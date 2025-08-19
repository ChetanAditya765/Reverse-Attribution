# ra.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients


def _to_numpy_tree(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        T = type(x)
        return T(_to_numpy_tree(v) for v in x)
    return x


def _extract_logits(out: Any) -> torch.Tensor:
    return out.logits if hasattr(out, "logits") else out


class ReverseAttribution:
    def __init__(
        self,
        model: nn.Module,
        baseline: Optional[Union[torch.Tensor, str]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.baseline = baseline

    def _input_kind(self, x: torch.Tensor) -> str:
        if x.dim() == 3 and torch.is_floating_point(x):
            return "text_emb"  # (B, L, D)
        if x.dim() == 4 and torch.is_floating_point(x):
            return "image"     # (B, C, H, W)
        if not torch.is_floating_point(x):
            return "ids"
        return "other"

    def _forward_logits(self, x: torch.Tensor, add_args: Optional[tuple]) -> torch.Tensor:
        if add_args:
            out = self.model(x, *add_args)
        else:
            out = self.model(x)
        return _extract_logits(out)

    def _ig_attribute(
        self,
        x: torch.Tensor,
        target: int,
        n_steps: int,
        baselines: Optional[torch.Tensor],
        add_args: Optional[tuple],
    ) -> torch.Tensor:
        ig = IntegratedGradients(self.model)
        attr = ig.attribute(
            x,
            target=target,
            n_steps=n_steps,
            baselines=baselines,
            additional_forward_args=add_args,
        )
        return attr

    def _prepare_baseline(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        if isinstance(self.baseline, torch.Tensor):
            b = self.baseline.to(x.device)
            if b.shape != x.shape:
                b = b.expand_as(x)
            return b
        if kind == "text_emb":
            return torch.zeros_like(x)
        if kind == "image":
            return torch.zeros_like(x)
        return torch.zeros_like(x)

    def _mask_token(self, x: torch.Tensor, token_idx: int, baseline: torch.Tensor) -> torch.Tensor:
        x_m = x.clone()
        if x_m.dim() == 3:
            x_m[:, token_idx, :] = baseline[:, token_idx, :]
        else:
            xf = x_m.view(x_m.size(0), -1)
            bf = baseline.view(baseline.size(0), -1)
            if token_idx < xf.size(1):
                xf[:, token_idx] = bf[:, token_idx]
            x_m = xf.view_as(x)
        return x_m

    def _mask_pixel(self, x: torch.Tensor, h: int, w: int, baseline: torch.Tensor) -> torch.Tensor:
        x_m = x.clone()
        if x_m.dim() == 4:
            x_m[:, :, h, w] = baseline[:, :, h, w]
        else:
            xf = x_m.view(x_m.size(0), -1)
            bf = baseline.view(baseline.size(0), -1)
            idx = h
            if idx < xf.size(1):
                xf[:, idx] = bf[:, idx]
            x_m = xf.view_as(x)
        return x_m

    def explain(
        self,
        x: torch.Tensor,
        y_true: int,
        top_m: int = 5,
        n_steps: int = 50,
        additional_forward_args: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        x = x.to(self.device)
        kind = self._input_kind(x)
        with torch.no_grad():
            logits = self._forward_logits(x, additional_forward_args)
            probs = F.softmax(logits, dim=-1)
            y_hat = int(torch.argmax(logits, dim=-1).item())
            _, top2 = torch.topk(probs, 2, dim=-1)
            runner_up = int(top2[0, 1].item())

        baselines = self._prepare_baseline(x, kind)
        phi = self._ig_attribute(x, target=y_hat, n_steps=n_steps, baselines=baselines, add_args=additional_forward_args)

        if kind == "text_emb" and phi.dim() == 3:
            token_scores = phi.abs().sum(-1)[0]
            neg_token_idx = torch.nonzero(token_scores > 0, as_tuple=False).view(-1)
            # Use signed token sums to get negative tokens
            signed = phi.sum(-1)[0]
            neg_token_idx = torch.nonzero(signed < 0, as_tuple=False).view(-1)
            idx_list = neg_token_idx.detach().cpu().tolist()
        elif kind == "image" and phi.dim() in (3, 4):
            if phi.dim() == 4:
                per_pixel = phi.abs().sum(1)[0]
                signed = phi.sum(1)[0]
            else:
                per_pixel = phi.abs()[0]
                signed = phi[0]
            neg_positions = torch.nonzero(signed < 0, as_tuple=False)
            idx_list = [(int(h.item()), int(w.item())) for h, w in neg_positions]
        else:
            flat = phi.view(-1)
            idx_list = torch.nonzero(flat < 0, as_tuple=False).view(-1).detach().cpu().tolist()

        if len(idx_list) == 0:
            return {
                "counter_evidence": [],
                "a_flip": 0.0,
                "phi": _to_numpy_tree(phi),
                "y_hat": y_hat,
                "runner_up": runner_up,
            }

        original_prob = float(probs[0, y_true].item())
        deltas: List[float] = []
        limit = min(len(idx_list), max(1, top_m * 4))

        for k in range(limit):
            idx = idx_list[k]
            if kind == "text_emb":
                x_masked = self._mask_token(x, idx, baselines)
            elif kind == "image":
                if isinstance(idx, tuple):
                    h, w = idx
                else:
                    h = int(idx)
                    w = 0
                x_masked = self._mask_pixel(x, h, w, baselines)
            else:
                xf = x.clone().view(1, -1)
                bf = baselines.view(1, -1)
                if isinstance(idx, tuple):
                    flat_i = idx[0]
                else:
                    flat_i = int(idx)
                if flat_i < xf.size(1):
                    xf[0, flat_i] = bf[0, flat_i]
                x_masked = xf.view_as(x)

            with torch.no_grad():
                m_logits = self._forward_logits(x_masked, additional_forward_args)
                m_probs = F.softmax(m_logits, dim=-1)
                m_prob = float(m_probs[0, y_true].item())
            deltas.append(m_prob - original_prob)

        entries: List[Tuple] = []
        if kind == "text_emb":
            for idx, d in zip(idx_list[:limit], deltas):
                contrib = float(phi[0, idx].sum().detach().cpu().item())
                entries.append((int(idx), contrib, float(d)))
        elif kind == "image":
            if phi.dim() == 4:
                per_pixel_signed = phi.sum(1)[0]
            else:
                per_pixel_signed = phi[0]
            for pos, d in zip(idx_list[:limit], deltas):
                h, w = (pos if isinstance(pos, tuple) else (int(pos), 0))
                contrib = float(per_pixel_signed[h, w].detach().cpu().item())
                entries.append(((h, w), contrib, float(d)))
        else:
            flat = phi.view(-1)
            for idx, d in zip(idx_list[:limit], deltas):
                contrib = float(flat[int(idx)].detach().cpu().item())
                entries.append((int(idx), contrib, float(d)))

        entries.sort(key=lambda t: abs(t[2]), reverse=True)
        entries = entries[:top_m]

        phi_runner = self._ig_attribute(
            x, target=runner_up, n_steps=n_steps, baselines=baselines, add_args=additional_forward_args
        )
        a_flip = float(0.5 * torch.sum(torch.abs(phi - phi_runner)).detach().cpu().item())

        return _to_numpy_tree(
            {
                "counter_evidence": entries,
                "a_flip": a_flip,
                "phi": phi,
                "y_hat": y_hat,
                "runner_up": runner_up,
            }
        )
