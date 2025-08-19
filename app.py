from __future__ import annotations
import json
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import math
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Reverse Attribution", page_icon="🧭", layout="wide")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

ReverseAttribution = None
ExplanationVisualizer = None
ExplainerHub = None
get_bert_model = None
get_resnet56_model = None
if "ra_demo" not in st.session_state:
    st.session_state.ra_demo = {
        "text": "This movie was fantastic with brilliant acting!",
        "tokens": [],          # list[str]
        "ids": None,           # torch.Tensor
        "attn": None,          # torch.Tensor or None
        "pred": None,          # int
        "probs": None,         # np.ndarray
        "ra_out": None,        # dict
        "mask_idxs": set(),    # selected token indices to mask
        "need_recompute": False
    }

def _import_ra_deps() -> None:
    global ReverseAttribution, ExplanationVisualizer, ExplainerHub, get_bert_model, get_resnet56_model
    if ReverseAttribution is None:
        try:
            from ra import ReverseAttribution as _RA
            ReverseAttribution = _RA
        except Exception:
            ReverseAttribution = None
    if ExplanationVisualizer is None:
        try:
            from ra.visualizer import ExplanationVisualizer as _Viz
            ExplanationVisualizer = _Viz
        except Exception:
            try:
                from visualizer import ExplanationVisualizer as _Viz
                ExplanationVisualizer = _Viz
            except Exception:
                ExplanationVisualizer = None
    if ExplainerHub is None:
        try:
            from ra.explainer_utils import ExplainerHub as _Hub
            ExplainerHub = _Hub
        except Exception:
            try:
                from ra.explainer_utils import ExplainerHub as _Hub
                ExplainerHub = _Hub
            except Exception:
                ExplainerHub = None
    if get_bert_model is None or get_resnet56_model is None:
        try:
            from models import get_bert_model as _gb, get_resnet56_model as _gr
            get_bert_model = _gb
            get_resnet56_model = _gr
        except Exception:
            pass


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _flex_load_state_dict(model: nn.Module, ckpt_path: Path) -> None:
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict):
        for key in ["state_dict", "model_state_dict", "net", "model"]:
            if key in sd and isinstance(sd[key], dict):
                sd = sd[key]
                break
    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()


def _load_history_json(dirpath: Path) -> Optional[Any]:
    f = dirpath / "training_history.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _best_from_history(hist: Any) -> Dict[str, Any]:
    if hist is None:
        return {}
    if isinstance(hist, dict) and "val_accuracy" in hist and isinstance(hist["val_accuracy"], list):
        epochs = len(hist.get("val_accuracy", []))
        vals = [{
            "val_accuracy": hist.get("val_accuracy", [None]*epochs)[i],
            "val_loss": hist.get("val_loss", [None]*epochs)[i],
            "train_accuracy": hist.get("train_accuracy", [None]*epochs)[i],
            "train_loss": hist.get("train_loss", [None]*epochs)[i],
        } for i in range(epochs)]
    elif isinstance(hist, list):
        vals = hist
        epochs = len(vals)
    else:
        return {}

    def _get_best(metric_name: str, bigger_is_better: bool = True):
        seq = [(i+1, ep.get(metric_name)) for i, ep in enumerate(vals) if isinstance(ep, dict)]
        seq = [(ep, v) for ep, v in seq if isinstance(v, (int, float))]
        if not seq:
            return None, None
        idx, val = (max if bigger_is_better else min)(seq, key=lambda t: t[1])
        return val, idx

    best_val_acc, best_epoch_acc = _get_best("val_accuracy", True)
    best_val_loss, best_epoch_loss = _get_best("val_loss", False)
    last = vals[-1] if vals else {}
    out = {"epochs": epochs}
    if best_val_acc is not None:
        out["best/val_accuracy"] = float(best_val_acc)
        out["best/val_accuracy_epoch"] = int(best_epoch_acc)
    if best_val_loss is not None:
        out["best/val_loss"] = float(best_val_loss)
        out["best/val_loss_epoch"] = int(best_epoch_loss)
    if isinstance(last.get("train_accuracy"), (int, float)):
        out["final/train_accuracy"] = float(last["train_accuracy"])
    if isinstance(last.get("val_accuracy"), (int, float)):
        out["final/val_accuracy"] = float(last["val_accuracy"])
    return out


def _read_ext_results(ckpt_root: Path, subdir: str) -> Dict[str, Any]:
    keys_map = {"bert_imdb": "imdb_results", "roberta_yelp": "yelp_results", "resnet56_cifar10": "cifar10_results"}
    target_key = keys_map.get(subdir)
    for candidate in [ckpt_root / "comprehensive_evaluation_results.json",
                      ckpt_root / subdir / "comprehensive_evaluation_results.json"]:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                block = data.get(target_key, data)
                out: Dict[str, Any] = {}
                std = block.get("standard_metrics", {})
                ra = block.get("ra_analysis", {}).get("summary", {})
                if isinstance(std.get("avg_loss"), (int, float)):
                    out["avg_loss"] = float(std["avg_loss"])
                if isinstance(std.get("num_samples"), (int, float)):
                    out["num_samples"] = int(std["num_samples"])
                if isinstance(ra.get("avg_a_flip"), (int, float)):
                    out["a_flip"] = float(ra["avg_a_flip"])
                return out
            except Exception:
                return {}
    return {}


def _eval_text_auto(model, dataset: str, max_samples: int, batch_size: int) -> Dict[str, float]:
    texts, labels = [], []
    try:
        from torchtext.datasets import IMDB, YelpReviewPolarity
        if dataset == "imdb":
            try: it = IMDB(split="test")
            except TypeError: _, it = IMDB()
            label_map = {"pos": 1, "neg": 0}
            for lbl, txt in it:
                texts.append(txt); labels.append(label_map.get(lbl, 0))
                if len(texts) >= max_samples: break
        else:
            try: it = YelpReviewPolarity(split="test")
            except TypeError: _, it = YelpReviewPolarity()
            for lbl, txt in it:
                texts.append(txt); labels.append(0 if int(lbl)==1 else 1)
                if len(texts) >= max_samples: break
    except Exception:
        from datasets import load_dataset
        stream_split = "test"
        it = load_dataset("imdb" if dataset == "imdb" else "yelp_polarity", split=stream_split, streaming=True)
        for ex in it:
            texts.append(ex["text"]); labels.append(int(ex["label"]))
            if len(texts) >= max_samples: break

    model.eval()
    probs_all: List[np.ndarray] = []
    losses: List[float] = []
    labels_torch_all: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = model.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        y = torch.tensor(labels[i:i+batch_size], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits if hasattr(out, "logits") else out
            losses.append(float(F.cross_entropy(logits, y)))
        probs_all.append(F.softmax(logits, dim=1).detach().cpu().numpy())
        labels_torch_all.append(y.detach().cpu())
    probs = np.concatenate(probs_all, axis=0)
    labels_np = torch.cat(labels_torch_all, dim=0).numpy()
    return {"Accuracy": float((probs.argmax(axis=1) == labels_np).mean()), "Avg CE Loss": float(np.mean(losses))}


@st.cache_resource(show_spinner=False)
def load_bert_imdb(ckpt_root: Path, num_labels: int = 2):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = "bert-base-uncased"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    ck = ckpt_root / "bert_imdb" / "best_model.pt"
    if ck.exists():
        _flex_load_state_dict(mdl, ck)
    mdl.tokenizer = tok
    return mdl


@st.cache_resource(show_spinner=False)
def load_roberta_yelp(ckpt_root: Path, num_labels: int = 2):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = "roberta-base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name, num_labels=num_labels)
    ck = ckpt_root / "roberta_yelp" / "best_model.pt"
    if ck.exists():
        _flex_load_state_dict(mdl, ck)
    mdl.tokenizer = tok
    return mdl


def encode_text(model, text: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not hasattr(model, "tokenizer"):
        raise RuntimeError("Tokenizer missing on model.")
    enc = model.tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    ids = enc["input_ids"].to(DEVICE)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(DEVICE)
    return ids, mask


def forward_logits_text(model, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]):
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.long()
    out = model(input_ids=input_ids, attention_mask=attn_mask) if attn_mask is not None else model(input_ids=input_ids)
    return out.logits if hasattr(out, "logits") else out


def _make_resnet56(num_classes: int = 10):
    try:
        from models.resnet_cifar import resnet56_cifar
        return resnet56_cifar(num_classes=num_classes)
    except Exception:
        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
                    nn.Linear(256, num_classes),
                )
            def forward(self, x):
                return self.net(x)
        return TinyCNN()


@st.cache_resource(show_spinner=False)
def load_resnet56_cifar10(ckpt_root: Path, num_classes: int = 10):
    mdl = _make_resnet56(num_classes=num_classes)
    ck = ckpt_root / "resnet56_cifar10" / "best_model.pt"
    if ck.exists():
        _flex_load_state_dict(mdl, ck)
    else:
        mdl.to(DEVICE).eval()
    return mdl


def img32_from_uploader(file) -> torch.Tensor:
    img = Image.open(file).convert("RGB").resize((32, 32))
    arr = (np.array(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)


def _find_embedding_layer(m) -> Optional[nn.Embedding]:
    if hasattr(m, "get_input_embeddings"):
        try:
            e = m.get_input_embeddings()
            if hasattr(e, "weight"):
                return e
        except Exception:
            pass
    for name in ["bert", "roberta", "distilbert", "albert", "xlm_roberta", "model", "base_model", "encoder", "backbone", "transformer"]:
        sub = getattr(m, name, None)
        if sub is None:
            continue
        if hasattr(sub, "get_input_embeddings"):
            try:
                e = sub.get_input_embeddings()
                if hasattr(e, "weight"):
                    return e
            except Exception:
                pass
        if hasattr(sub, "embeddings"):
            emb = getattr(sub, "embeddings")
            if hasattr(emb, "word_embeddings"):
                we = getattr(emb, "word_embeddings")
                if hasattr(we, "weight"):
                    return we
    for _, mod in m.named_modules():
        if isinstance(mod, nn.Embedding):
            return mod
    return None


def run_baseline_captum(model, model_type: str, payload: Any, target: Optional[int] = None, add_args: Optional[tuple] = None) -> Dict[str, Any]:
    from captum.attr import LayerIntegratedGradients, IntegratedGradients
    model.eval()
    if model_type.startswith("bert") or model_type.startswith("text") or (isinstance(payload, dict) and "input_ids" in payload):
        if isinstance(payload, dict):
            input_ids = payload["input_ids"].to(DEVICE)
            if torch.is_floating_point(input_ids):
                input_ids = input_ids.long()
            attn = payload.get("attention_mask")
            attn = attn.to(DEVICE) if attn is not None else None
        else:
            input_ids = payload
            if torch.is_floating_point(input_ids):
                input_ids = input_ids.long()
            attn = add_args[0] if add_args else None
        emb_layer = _find_embedding_layer(model)
        if emb_layer is None:
            raise RuntimeError("No embedding layer found for text IG.")
        def fwd_ids(ids, attention_mask=None):
            out = model(input_ids=ids, attention_mask=attention_mask) if attention_mask is not None else model(input_ids=ids)
            return out.logits if hasattr(out, "logits") else out
        lig = LayerIntegratedGradients(fwd_ids, emb_layer)
        attributions = lig.attribute(inputs=input_ids, additional_forward_args=(attn,), target=target, n_steps=50)
        arr = attributions.detach().cpu().numpy() if isinstance(attributions, torch.Tensor) else np.array(attributions)
        return {"captum_lig_embeddings": arr}
    x = payload
    ig = IntegratedGradients(model)
    attributions = ig.attribute(inputs=x, target=target, additional_forward_args=add_args)
    arr = attributions.detach().cpu().numpy() if isinstance(attributions, torch.Tensor) else np.array(attributions)
    return {"captum_ig_image": arr}


def _prepare_text_for_ra(model, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]):
    emb_layer = _find_embedding_layer(model)
    if emb_layer is None:
        raise RuntimeError("No embedding layer found for RA.")
    ids = input_ids.long()
    with torch.no_grad():
        embeds = emb_layer(ids)
    embeds = embeds.detach().requires_grad_(True)
    class RAEmbedsAdapter(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, inputs_embeds, attention_mask=None):
            out = self.base(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return out.logits if hasattr(out, "logits") else out
    return RAEmbedsAdapter(model), embeds, (attn_mask,)


def _detach_tree(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: _detach_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        T = type(x)
        return T(_detach_tree(v) for v in x)
    return x


def run_ra(model, input_tensor: torch.Tensor, y_true: Optional[int] = None, add_args: Optional[tuple] = None) -> Dict[str, Any]:
    if ReverseAttribution is None:
        raise RuntimeError("ReverseAttribution not available.")
    is_text_like = (isinstance(input_tensor, torch.Tensor) and not torch.is_floating_point(input_tensor)) or (
        isinstance(add_args, tuple) and len(add_args) > 0 and isinstance(add_args[0], torch.Tensor)
    )
    if is_text_like:
        attn = add_args[0] if (isinstance(add_args, tuple) and len(add_args) > 0) else None
        adapted_model, embeds, fwd_args = _prepare_text_for_ra(model, input_tensor, attn)
        ra = ReverseAttribution(adapted_model, device=DEVICE)
        return ra.explain(embeds, y_true=y_true, additional_forward_args=fwd_args)
    class LogitsOnly(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def forward(self, *args, **kwargs):
            out = self.inner(*args, **kwargs)
            return out.logits if hasattr(out, "logits") else out
    wrapped = LogitsOnly(model)
    ra = ReverseAttribution(wrapped, device=DEVICE)
    return ra.explain(input_tensor, y_true=y_true, additional_forward_args=add_args)


def _merge_wp_tokens(tokens: List[str], scores: List[float]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for t, s in zip(tokens, scores):
        if t in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            continue
        s = abs(float(s))
        if t.startswith("##") and out:
            pt, ps = out[-1]
            out[-1] = (pt + t[2:], ps + s)
        else:
            out.append((t, s))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def show_text_attrib_table(model, input_ids: torch.Tensor, base: Dict[str, Any], tokenizer=None, attn_mask: Optional[torch.Tensor] = None, top_k: int = 15) -> None:
    arr = None
    for k in ("captum_lig_embeddings", "captum_ig_embeddings", "captum_attribution", "result", "phi"):
        if isinstance(base, dict) and k in base:
            arr = base[k]
            break
    if arr is None:
        st.write(base); return
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    elif not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 3:
        scores = np.sum(np.abs(arr), axis=-1)[0]
    elif arr.ndim == 2:
        scores = np.sum(np.abs(arr), axis=-1)
    else:
        scores = np.abs(arr)
    ids = input_ids[0].detach().cpu().tolist()
    if attn_mask is not None:
        maskv = attn_mask[0].detach().cpu().numpy().astype(bool)
        if len(maskv) == len(scores):
            scores = scores[maskv]
            ids = [ids[i] for i in range(len(ids)) if maskv[i]]
    if tokenizer is None and hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    toks = tokenizer.convert_ids_to_tokens(ids) if tokenizer is not None else [str(i) for i in ids]
    merged = _merge_wp_tokens(toks, scores.tolist() if isinstance(scores, np.ndarray) else list(scores))
    merged = merged[:top_k]
    st.dataframe({"token": [t for t, _ in merged], "score": [float(v) for _, v in merged]}, use_container_width=True)


def show_image_heatmap(base: Dict[str, Any]) -> None:
    arr = None
    for k in ("captum_ig_image", "captum_attribution", "result", "phi"):
        if isinstance(base, dict) and k in base:
            arr = base[k]
            break
    if arr is None:
        st.write(base); return
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    elif not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.ndim == 4:
        heat = np.mean(np.abs(arr), axis=1)[0]
    elif arr.ndim == 3:
        heat = np.mean(np.abs(arr), axis=0)
    else:
        heat = np.abs(arr)
    heat = (heat - heat.min()) / (heat.ptp() + 1e-8)
    st.image(heat, caption="Attribution heatmap", use_column_width=True)


def metric_cards(items: Dict[str, Any]):
    if not items:
        return
    keys = list(items.keys())
    cols = st.columns(max(1, min(4, len(keys))))
    for i, k in enumerate(keys):
        with cols[i % len(cols)]:
            st.metric(k, f"{items[k]}")


# --- Interactive masking helpers (TEXT) ---
def _token_scores_from_attr(attr, attn_mask=None):
    if isinstance(attr, torch.Tensor):
        arr = attr.detach().cpu().numpy()
    else:
        arr = np.array(attr)
    if arr.ndim == 3:
        s = np.sum(np.abs(arr), axis=-1)[0]
    elif arr.ndim == 2:
        s = np.sum(np.abs(arr), axis=-1)
    else:
        s = np.abs(arr)
    if attn_mask is not None:
        m = attn_mask[0].detach().cpu().numpy().astype(bool)
        s = s[: len(m)][m]
    return s
@st.cache_data(show_spinner=False)
def _load_csv_anywhere(ckpt_root: Path, filename: str) -> Optional[pd.DataFrame]:
    # direct
    for p in [ckpt_root / filename, ckpt_root / "reports" / filename]:
        if p.exists():
            return pd.read_csv(p)
    # recursive (safe on Pathlib; avoids the ** fsspec issue)
    for p in ckpt_root.rglob(filename):
        try:
            return pd.read_csv(p)
        except Exception:
            continue
    return None
def _toggle_mask(i: int):
    s = st.session_state.ra_demo
    if i in s["mask_idxs"]:
        s["mask_idxs"].remove(i)
    else:
        s["mask_idxs"].add(i)
    s["need_recompute"] = True

@st.cache_data(show_spinner=False)
def _cached_logits(model_name: str, text: str) -> Tuple[np.ndarray, int, list, list]:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = None
    if model_name == "bert":
        model = load_bert_imdb(ckpt_root)
        tok = model.tokenizer
    else:
        model = load_roberta_yelp(ckpt_root)
        tok = model.tokenizer
    enc = tok([text], return_tensors="pt", truncation=True, padding=True)
    ids = enc["input_ids"].to(DEVICE)
    attn = enc.get("attention_mask")
    if attn is not None: attn = attn.to(DEVICE)
    with torch.no_grad():
        logits = forward_logits_text(model, ids, attn)
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred = int(np.argmax(probs))
    toks = tok.convert_ids_to_tokens(ids[0].detach().cpu().tolist())
    return probs, pred, toks, ids[0].detach().cpu().tolist()

def _mask_text_ids(ids: torch.Tensor, mask_idxs: set, tokenizer):
    ids = ids.clone()
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        # fallback: replace with [PAD] if MASK not available
        mask_id = tokenizer.pad_token_id
    for i in mask_idxs:
        if 0 <= i < ids.shape[1]:
            ids[0, i] = mask_id
    return ids

def _safe_scatter(df: pd.DataFrame, x: str, y: str, title: str):
    if x in df.columns and y in df.columns:
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y], s=12)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        st.pyplot(fig)


def _mask_tokens(ids: torch.Tensor, idxs: List[int], mask_id: int) -> torch.Tensor:
    ids2 = ids.clone()
    for i in idxs:
        if 0 <= i < ids2.size(1):
            ids2[0, i] = mask_id
    return ids2


def _render_token_picker(tokens: List[str], scores: np.ndarray, top_k: int = 15) -> List[int]:
    order = np.argsort(scores)[::-1]
    order = order[: min(top_k, len(order))]
    df = {
        "idx": [int(i) for i in order],
        "token": [tokens[i] for i in order],
        "score": [float(scores[i]) for i in order],
    }
    st.dataframe(df, hide_index=True, use_container_width=True)
    picked_idxs = st.multiselect("Click to mask tokens", options=order.tolist(), format_func=lambda i: f"{tokens[i]} (#{i})")
    return [int(i) for i in picked_idxs]

def _iter_text_samples(dataset_name: str, max_samples: int):
    """Yield (text, label) pairs without any '**' globbing."""
    n = 0
    try:
        # torchtext first (fast, no glob)
        from torchtext.datasets import IMDB, YelpReviewPolarity
        if dataset_name == "imdb":
            try: it = IMDB(split="test")
            except TypeError: _, it = IMDB()
            label_map = {"pos": 1, "neg": 0}
            for lbl, txt in it:
                yield txt, int(label_map.get(lbl, 0)); n += 1
                if n >= max_samples: break
        else:
            try: it = YelpReviewPolarity(split="test")
            except TypeError: _, it = YelpReviewPolarity()
            for lbl, txt in it:
                yield txt, (0 if int(lbl) == 1 else 1); n += 1
                if n >= max_samples: break
    except Exception:
        # HF datasets in streaming mode (no '**' patterns)
        from datasets import load_dataset
        dsname = "imdb" if dataset_name == "imdb" else "yelp_polarity"
        it = load_dataset(dsname, split="test", streaming=True)
        for ex in it:
            yield ex["text"], int(ex["label"]); n += 1
            if n >= max_samples: break

def _entropy_np(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    H = -np.sum(p * np.log(p))
    # normalize by ln(C) to keep it ~[0,1]
    return float(H / math.log(len(p)))

def _runner_up_idx(probs: np.ndarray) -> int:
    return int(np.argsort(-probs)[1])

def _a_flip_from_captum_text(model, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor], pred_idx: int, runner_idx: int) -> float:
    """Fallback A-Flip using LayerIntegratedGradients on embeddings for pred vs. runner."""
    from captum.attr import LayerIntegratedGradients
    emb_layer = _find_embedding_layer(model)
    if emb_layer is None:
        return float("nan")

    def fwd_ids(ids, attention_mask=None):
        out = model(input_ids=ids, attention_mask=attention_mask) if attention_mask is not None else model(input_ids=ids)
        return out.logits if hasattr(out, "logits") else out

    lig = LayerIntegratedGradients(fwd_ids, emb_layer)
    ids = input_ids.long()
    add_args = (attn_mask,) if attn_mask is not None else (None,)

    A = lig.attribute(inputs=ids, target=pred_idx, additional_forward_args=add_args, n_steps=24)
    B = lig.attribute(inputs=ids, target=runner_idx, additional_forward_args=add_args, n_steps=24)

    A = A.detach().cpu().numpy()
    B = B.detach().cpu().numpy()
    if A.ndim == 3: A = np.sum(np.abs(A), axis=-1)[0]
    if B.ndim == 3: B = np.sum(np.abs(B), axis=-1)[0]
    return float(0.5 * np.sum(np.abs(A - B)))

def _compute_a_flip_text(model, ids: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[float, Optional[int]]:
    """Try RA first; fallback to Captum A-Flip if RA unavailable."""
    _import_ra_deps()
    with torch.no_grad():
        logits = forward_logits_text(model, ids, mask)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred = int(np.argmax(probs))
    runner = _runner_up_idx(probs)

    try:
        ra_out = run_ra(model, ids, y_true=pred, add_args=(mask,))
        a = ra_out.get("a_flip") or ra_out.get("A-Flip")
        return float(a), runner
    except Exception:
        pass

    try:
        a = _a_flip_from_captum_text(model, ids, mask, pred, runner)
        return float(a), runner
    except Exception:
        return float("nan"), runner

# --- Interactive masking helpers (IMAGE) ---
def _patch_coords(h=32, w=32, ph=8, pw=8):
    coords = []
    pid = 0
    for y in range(0, h, ph):
        for x in range(0, w, pw):
            coords.append((pid, y, y+ph, x, x+pw))
            pid += 1
    return coords


def _mask_patches(x: torch.Tensor, picks: List[int], mode="mean", ph=8, pw=8):
    x2 = x.clone()
    _, _, H, W = x2.size()
    coords = _patch_coords(H, W, ph, pw)
    if mode == "mean":
        val = float(x2.mean().item())
    else:
        val = 0.0
    for pid, y0, y1, x0, x1 in coords:
        if pid in picks:
            x2[:, :, y0:y1, x0:x1] = val
    return x2


def _render_patch_picker(ph=8, pw=8):
    pids = [pid for pid, *_ in _patch_coords(32, 32, ph, pw)]
    pick = st.multiselect("Mask patches (8×8 grid)", options=pids, format_func=lambda i: f"#{i}")
    return [int(i) for i in pick]


with st.sidebar:
    st.title("Controls")
    ckpt_root = Path(st.text_input("Checkpoints root", value=str((ROOT / "checkpoints").resolve())))
    task = st.selectbox("Task", ["Text: BERT (IMDB)", "Text: RoBERTa (Yelp)", "Vision: ResNet56 (CIFAR-10)"])
    mode = st.radio("Mode", ["Overview", "Live Demo", "RA vs Baselines", "Evaluate", "Analysis", "Reproducibility"], index=0)
    top_k = st.slider("Top-K tokens", 3, 30, 15)
    show_interactive = st.checkbox("Interactive visuals (if available)", True)
    st.caption(f"Device: **{DEVICE.upper()}**")

st.title("Reverse Attribution")

if mode == "Overview":
    st.markdown(
        """
**Goal.** Show how Reverse Attribution (RA) highlights *counter-evidence* that flips predictions, alongside standard baselines.

**What you can do here:**
- Run **Live Demo** on text or CIFAR-10.
- Compare **RA vs Captum/LIG**.
- Run a mini **Evaluate** with Accuracy / Avg. Loss.
- Load your trained **checkpoints** directly.
"""
    )

elif mode == "Live Demo":
    # ---------- Session state init ----------
    if "demo_text_bert" not in st.session_state:
        st.session_state.demo_text_bert = {
            "text": "This movie was fantastic with brilliant acting!",
            "ids": None, "attn": None, "tokens": [],
            "pred": None, "probs": None,
            "mask_idxs": set(),
            "ra_out": None,
        }
    if "demo_text_roberta" not in st.session_state:
        st.session_state.demo_text_roberta = {
            "text": "The service was quick and the food was amazing!",
            "ids": None, "attn": None, "tokens": [],
            "pred": None, "probs": None,
            "mask_idxs": set(),
            "ra_out": None,
        }
    if "demo_vision" not in st.session_state:
        st.session_state.demo_vision = {
            "x": None, "pred": None, "probs": None,
            "mask_grid": set(),  # set of (r,c)
            "ra_out": None,
            "ph": 8, "pw": 8,
        }

    # ---------- Small utilities local to this block ----------
    def _special(tok: str) -> bool:
        return tok in ("[CLS]", "[SEP]", "[PAD]", "[UNK]")

    def _apply_token_mask(ids: torch.Tensor, mask_idxs: set, tokenizer) -> torch.Tensor:
        out = ids.clone()
        mask_id = getattr(tokenizer, "mask_token_id", None) or getattr(tokenizer, "pad_token_id", 0)
        for i in mask_idxs:
            if 0 <= i < out.shape[1]:
                out[0, i] = mask_id
        return out

    def _recompute_text(model, ids, attn):
        with torch.no_grad():
            logits = forward_logits_text(model, ids, attn)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return probs, pred

    def _token_checkboxes(prefix: str, tokens: list[str], selected: set[int], ncols: int = 8) -> set[int]:
        cols = st.columns(ncols)
        new_sel = set()
        for i, tok in enumerate(tokens):
            if _special(tok):  # still clickable, but you may skip if you prefer
                label = f"{tok}"
            else:
                label = tok
            key = f"{prefix}_tok_{i}"
            with cols[i % ncols]:
                checked = st.checkbox(label, value=(i in selected), key=key)
            if checked:
                new_sel.add(i)
        return new_sel

    def _run_ra_once(model, ids_or_x, y_pred, attn=None):
        try:
            _import_ra_deps()
            add = (attn,) if attn is not None else None
            return run_ra(model, ids_or_x, y_true=y_pred, add_args=add)
        except Exception as e:
            st.caption(f"RA unavailable: {e}")
            return None

    # =========================
    # TEXT: BERT
    # =========================
    if task.startswith("Text: BERT"):
        model = load_bert_imdb(ckpt_root)
        s = st.session_state.demo_text_bert

        s["text"] = st.text_area("Input", s["text"], height=120, key="bert_text_area")

        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("Predict + Explain", key="bert_predict"):
                ids, mask = encode_text(model, s["text"])
                with torch.no_grad():
                    logits = forward_logits_text(model, ids, mask)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                pred = int(np.argmax(probs))
                toks = model.tokenizer.convert_ids_to_tokens(ids[0].detach().cpu().tolist())
                # store
                s.update({"ids": ids, "attn": mask, "tokens": toks, "pred": pred, "probs": probs,
                          "mask_idxs": set(), "ra_out": None})
        with colB:
            if st.button("Apply mask", key="bert_applymask") and s["ids"] is not None:
                masked_ids = _apply_token_mask(s["ids"], s["mask_idxs"], model.tokenizer)
                probs2, pred2 = _recompute_text(model, masked_ids, s["attn"])
                delta = float(probs2[s["pred"]] - s["probs"][s["pred"]])
                st.info(f"Δprob (class {s['pred']}): {delta:+.4f}  |  New Prob={probs2[s['pred']]:.2f}")
                # RA recompute (optional)
                ra2 = _run_ra_once(model, masked_ids, s["pred"], s["attn"])
                if isinstance(ra2, dict) and isinstance(s["ra_out"], dict):
                    a1 = ra2.get("a_flip"); a0 = s["ra_out"].get("a_flip")
                    if isinstance(a0, (int, float)) and isinstance(a1, (int, float)):
                        st.caption(f"A-Flip: {a0:.4f} → {a1:.4f} (masked)")
                # keep masked prediction/probs separate; do not overwrite original
                st.session_state["bert_last_mask_probs"] = probs2
                st.session_state["bert_last_mask_pred"] = pred2
        with colC:
            if st.button("Reset masks", key="bert_resetmask") and s["tokens"]:
                # clear UI checkboxes
                for i in range(len(s["tokens"])):
                    k = f"bert_tok_{i}"
                    if k in st.session_state:
                        st.session_state[k] = False
                s["mask_idxs"] = set()

        # Show current prediction
        if s["ids"] is not None:
            st.success(f"Pred: {s['pred']} • Prob={s['probs'][s['pred']]:.2f}")

            st.caption("Click tokens to mask/unmask:")
            new_sel = _token_checkboxes("bert", s["tokens"], s["mask_idxs"], ncols=8)
            if new_sel != s["mask_idxs"]:
                s["mask_idxs"] = new_sel

            # Compute RA once (for unmasked)
            if s["ra_out"] is None:
                s["ra_out"] = _run_ra_once(model, s["ids"], s["pred"], s["attn"])

            # Visualize RA or fallback
            if isinstance(s["ra_out"], dict):
                show_text_attrib_table(model, s["ids"], s["ra_out"], tokenizer=model.tokenizer, attn_mask=s["attn"], top_k=top_k)
                a_flip = s["ra_out"].get("a_flip")
                runner = s["ra_out"].get("runner_up") or s["ra_out"].get("runner_up_idx")
                if isinstance(a_flip, (int, float)):
                    st.caption(f"A-Flip: {a_flip:.4f} • Pred={s['pred']}" + (f" • Runner-up={int(runner)}" if runner is not None else ""))

    # =========================
    # TEXT: RoBERTa
    # =========================
    elif task.startswith("Text: RoBERTa"):
        model = load_roberta_yelp(ckpt_root)
        s = st.session_state.demo_text_roberta

        s["text"] = st.text_area("Input", s["text"], height=120, key="roberta_text_area")

        colA, colB, colC = st.columns([1,1,1])
        with colA:
            if st.button("Predict + Explain", key="roberta_predict"):
                ids, mask = encode_text(model, s["text"])
                with torch.no_grad():
                    logits = forward_logits_text(model, ids, mask)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                pred = int(np.argmax(probs))
                toks = model.tokenizer.convert_ids_to_tokens(ids[0].detach().cpu().tolist())
                s.update({"ids": ids, "attn": mask, "tokens": toks, "pred": pred, "probs": probs,
                          "mask_idxs": set(), "ra_out": None})
        with colB:
            if st.button("Apply mask", key="roberta_applymask") and s["ids"] is not None:
                masked_ids = _apply_token_mask(s["ids"], s["mask_idxs"], model.tokenizer)
                probs2, pred2 = _recompute_text(model, masked_ids, s["attn"])
                delta = float(probs2[s["pred"]] - s["probs"][s["pred"]])
                st.info(f"Δprob (class {s['pred']}): {delta:+.4f}  |  New Prob={probs2[s['pred']]:.2f}")
                ra2 = _run_ra_once(model, masked_ids, s["pred"], s["attn"])
                if isinstance(ra2, dict) and isinstance(s["ra_out"], dict):
                    a1 = ra2.get("a_flip"); a0 = s["ra_out"].get("a_flip")
                    if isinstance(a0, (int, float)) and isinstance(a1, (int, float)):
                        st.caption(f"A-Flip: {a0:.4f} → {a1:.4f} (masked)")
                st.session_state["roberta_last_mask_probs"] = probs2
                st.session_state["roberta_last_mask_pred"] = pred2
        with colC:
            if st.button("Reset masks", key="roberta_resetmask") and s["tokens"]:
                for i in range(len(s["tokens"])):
                    k = f"roberta_tok_{i}"
                    if k in st.session_state:
                        st.session_state[k] = False
                s["mask_idxs"] = set()

        if s["ids"] is not None:
            st.success(f"Pred: {s['pred']} • Prob={s['probs'][s['pred']]:.2f}")

            st.caption("Click tokens to mask/unmask:")
            new_sel = _token_checkboxes("roberta", s["tokens"], s["mask_idxs"], ncols=8)
            if new_sel != s["mask_idxs"]:
                s["mask_idxs"] = new_sel

            if s["ra_out"] is None:
                s["ra_out"] = _run_ra_once(model, s["ids"], s["pred"], s["attn"])

            if isinstance(s["ra_out"], dict):
                show_text_attrib_table(model, s["ids"], s["ra_out"], tokenizer=model.tokenizer, attn_mask=s["attn"], top_k=top_k)
                a_flip = s["ra_out"].get("a_flip")
                runner = s["ra_out"].get("runner_up") or s["ra_out"].get("runner_up_idx")
                if isinstance(a_flip, (int, float)):
                    st.caption(f"A-Flip: {a_flip:.4f} • Pred={s['pred']}" + (f" • Runner-up={int(runner)}" if runner is not None else ""))

    # =========================
    # VISION: ResNet56 (CIFAR-10)
    # =========================
    else:
        model = load_resnet56_cifar10(ckpt_root)
        s = st.session_state.demo_vision

        img = st.file_uploader("Upload a 32×32 RGB image", type=["png", "jpg", "jpeg"])
        colA, colB, colC = st.columns([1,1,1])

        with colA:
            if img is not None and st.button("Predict + Explain", key="vision_predict"):
                x = img32_from_uploader(img)
                with torch.no_grad():
                    logits = model(x)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                pred = int(np.argmax(probs))
                s.update({"x": x, "pred": pred, "probs": probs, "mask_grid": set(), "ra_out": None})
        with colB:
            if st.button("Apply mask", key="vision_applymask") and s["x"] is not None:
                x = s["x"].clone()
                C, H, W = x.shape[1], x.shape[2], x.shape[3]
                ph, pw = s["ph"], s["pw"]
                for (r, c) in s["mask_grid"]:
                    r0, c0 = r*ph, c*pw
                    x[:, :, r0:r0+ph, c0:c0+pw] = x[:, :, r0:r0+ph, c0:c0+pw].mean()
                with torch.no_grad():
                    logits2 = model(x)
                probs2 = F.softmax(logits2, dim=1).detach().cpu().numpy()[0]
                delta = float(probs2[s["pred"]] - s["probs"][s["pred"]])
                st.info(f"Δprob (class {s['pred']}): {delta:+.4f}  |  New Prob={probs2[s['pred']]:.2f}")
                # RA recompute (optional)
                ra2 = _run_ra_once(model, x, s["pred"])
                if isinstance(ra2, dict) and isinstance(s["ra_out"], dict):
                    a1 = ra2.get("a_flip"); a0 = s["ra_out"].get("a_flip")
                    if isinstance(a0, (int, float)) and isinstance(a1, (int, float)):
                        st.caption(f"A-Flip: {a0:.4f} → {a1:.4f} (masked)")
                st.session_state["vision_last_mask_probs"] = probs2
        with colC:
            if st.button("Reset masks", key="vision_resetmask") and s["x"] is not None:
                # clear grid toggles
                ph, pw = s["ph"], s["pw"]
                H, W = 32, 32
                Rh, Cw = H // ph, W // pw
                for r in range(Rh):
                    for c in range(Cw):
                        k = f"vision_cell_{r}_{c}"
                        if k in st.session_state:
                            st.session_state[k] = False
                s["mask_grid"] = set()

        # Render current image + RA + patch grid
        if s["x"] is not None:
            st.success(f"Pred: {s['pred']} • Prob={s['probs'][s['pred']]:.2f}")

            st.image(
                (s["x"][0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8),
                caption="Input (32×32)", use_column_width=False
            )

            # Compute RA once (unmasked)
            if s["ra_out"] is None:
                s["ra_out"] = _run_ra_once(model, s["x"], s["pred"])

            if isinstance(s["ra_out"], dict):
                show_image_heatmap(s["ra_out"])
                a_flip = s["ra_out"].get("a_flip")
                if isinstance(a_flip, (int, float)):
                    st.caption(f"A-Flip: {a_flip:.4f} • Pred={s['pred']}")

            # Patch picker grid
            st.caption("Click patches to mask/unmask:")
            ph, pw = s["ph"], s["pw"]
            H, W = 32, 32
            Rh, Cw = H // ph, W // pw
            for r in range(Rh):
                cols = st.columns(Cw)
                for c in range(Cw):
                    key = f"vision_cell_{r}_{c}"
                    with cols[c]:
                        checked = st.checkbox(f"{r},{c}", value=((r, c) in s["mask_grid"]), key=key)
                    if checked:
                        s["mask_grid"].add((r, c))
                    elif (r, c) in s["mask_grid"]:
                        s["mask_grid"].remove((r, c))

elif mode == "RA vs Baselines":
    _import_ra_deps()
    left, right = st.columns(2)
    if task.startswith("Text"):
        model = load_bert_imdb(ckpt_root) if "BERT" in task else load_roberta_yelp(ckpt_root)
        text = st.text_area("Input", "The plot is thin but performances are superb.")
        ids, mask = encode_text(model, text)
        add = (mask,) if mask is not None else None
        with torch.no_grad():
            logits = forward_logits_text(model, ids, mask)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))
        with left:
            st.markdown("**Reverse Attribution**")
            try:
                ra_out = run_ra(model, ids, y_true=pred, add_args=add)
                if ExplanationVisualizer is not None:
                    viz = ExplanationVisualizer(str(ASSETS))
                    art = viz.visualize_ra_explanation(_detach_tree(ra_out), input_data=text, input_type="text", tokens=None, interactive=show_interactive)
                    if isinstance(art, dict) and art.get("heatmap") and Path(art["heatmap"]).exists():
                        st.image(art["heatmap"], use_column_width=True)
                    else:
                        show_text_attrib_table(model, ids, ra_out, tokenizer=getattr(model, "tokenizer", None), attn_mask=mask, top_k=top_k)
                else:
                    show_text_attrib_table(model, ids, ra_out, tokenizer=getattr(model, "tokenizer", None), attn_mask=mask, top_k=top_k)
                a_flip = ra_out.get("a_flip") or ra_out.get("A-Flip")
                runner = ra_out.get("runner_up") or ra_out.get("runner_up_idx") or ra_out.get("runner_up_class")
                if a_flip is not None or runner is not None:
                    st.caption(f"A-Flip: {float(a_flip):.4f} • Pred={pred}" + (f" • Runner-up={int(runner)}" if runner is not None else ""))
            except Exception as e:
                st.error(f"RA failed: {e}")
        with right:
            st.markdown("**Baselines (Captum LIG)**")
            try:
                base = run_baseline_captum(model, "text", {"input_ids": ids, "attention_mask": mask}, target=pred, add_args=add)
                show_text_attrib_table(model, ids, base, tokenizer=getattr(model, "tokenizer", None), attn_mask=mask, top_k=top_k)
            except Exception as e:
                st.error(f"Baseline failed: {e}")
    else:
        model = load_resnet56_cifar10(ckpt_root)
        img = st.file_uploader("Upload image (32×32)", type=["png", "jpg", "jpeg"])
        if img is not None:
            x = img32_from_uploader(img)
            with torch.no_grad():
                logits = model(x)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))
            with left:
                st.markdown("**Reverse Attribution**")
                try:
                    ra_out = run_ra(model, x, y_true=pred)
                    if ExplanationVisualizer is not None:
                        viz = ExplanationVisualizer(str(ASSETS))
                        art = viz.visualize_ra_explanation(_detach_tree(ra_out), input_data=x, input_type="image", tokens=None, interactive=show_interactive)
                        if isinstance(art, dict) and art.get("overlay") and Path(art["overlay"]).exists():
                            st.image(art["overlay"], use_column_width=True)
                        else:
                            show_image_heatmap(ra_out)
                    else:
                        show_image_heatmap(ra_out)
                    a_flip = ra_out.get("a_flip") or ra_out.get("A-Flip")
                    if a_flip is not None:
                        st.caption(f"A-Flip: {float(a_flip):.4f} • Pred={pred}")
                except Exception as e:
                    st.error(f"RA failed: {e}")
            with right:
                st.markdown("**Baselines (Captum LIG)**")
                try:
                    base = run_baseline_captum(model, "resnet_cifar", x, target=pred)
                    show_image_heatmap(base)
                except Exception as e:
                    st.error(f"Baseline failed: {e}")

elif mode == "Evaluate":
    st.subheader("Evaluation")
    eval_mode = st.radio("Choose mode", ["Saved Metrics (no compute)", "Quick Evaluation (compute)"], index=0, horizontal=True)

    if eval_mode.startswith("Saved"):
        if task.startswith("Text: BERT"):
            subdir, label = "bert_imdb", "BERT (IMDB)"
        elif task.startswith("Text: RoBERTa"):
            subdir, label = "roberta_yelp", "RoBERTa (Yelp)"
        else:
            subdir, label = "resnet56_cifar10", "ResNet56 (CIFAR-10)"

        st.markdown(f"**{label}**")
        hist = _load_history_json(ckpt_root / subdir)
        core = _best_from_history(hist)
        extra = _read_ext_results(ckpt_root, subdir)
        merged = {**core, **extra}
        if merged:
            show = {}
            if "best/val_accuracy" in merged: show["Best Val Acc"] = f"{merged['best/val_accuracy']:.4f}"
            if "best/val_loss" in merged:     show["Best Val Loss"] = f"{merged['best/val_loss']:.4f}"
            if "epochs" in merged:            show["Epochs"] = merged["epochs"]
            if "avg_loss" in merged:          show["Avg Loss"] = f"{merged['avg_loss']:.4f}"
            if "num_samples" in merged:       show["Samples"] = merged["num_samples"]
            if "a_flip" in merged:            show["A-Flip"] = f"{merged['a_flip']:.2f}"
            metric_cards(show)
            st.divider()
            st.caption(f"{label} • raw summary")
            st.json(merged)
        else:
            st.info("No recognizable metrics found. Add a training_history.json and/or comprehensive_evaluation_results.json.")

        with st.expander("Show full training_history.json"):
            p = ckpt_root / subdir / "training_history.json"
            if p.exists():
                try:
                    st.json(json.loads(p.read_text(encoding="utf-8")))
                except Exception as e:
                    st.warning(f"Could not parse: {e}")
            else:
                st.caption("Not found.")

    else:
        if task.startswith("Text: BERT"):
            model = load_bert_imdb(ckpt_root, num_labels=2)
            max_samples = st.slider("Max samples", 64, 2000, 64, step=64)
            batch_size = st.slider("Batch size", 4, 64, 16, step=4)
            with st.spinner("Evaluating IMDB…"):
                st.json(_eval_text_auto(model, "imdb", max_samples, batch_size))

        elif task.startswith("Text: RoBERTa"):
            model = load_roberta_yelp(ckpt_root, num_labels=2)
            max_samples = st.slider("Max samples", 64, 2000, 64, step=64)
            batch_size = st.slider("Batch size", 4, 64, 16, step=4)
            with st.spinner("Evaluating Yelp Polarity…"):
                st.json(_eval_text_auto(model, "yelp", max_samples, batch_size))

        else:
            import torchvision
            from torchvision import transforms
            from torch.utils.data import DataLoader

            model = load_resnet56_cifar10(ckpt_root, num_classes=10)
            max_samples = st.slider("Max samples", 64, 2000, 512, step=64)
            batch_size = st.slider("Batch size", 16, 256, 64, step=16)

            tfm = transforms.ToTensor()
            ds = torchvision.datasets.CIFAR10(root=str(ROOT / "data"), train=False, download=True, transform=tfm)
            if max_samples < len(ds):
                ds = torch.utils.data.Subset(ds, range(max_samples))
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

            with st.spinner("Evaluating CIFAR-10…"):
                logits_all, labels_all, losses = [], [], []
                model.eval()
                ce = nn.CrossEntropyLoss(reduction="mean")
                with torch.no_grad():
                    for x, y in dl:
                        x = x.to(DEVICE); y = y.to(DEVICE)
                        out = model(x)
                        losses.append(float(ce(out, y)))
                        logits_all.append(out.detach().cpu().numpy())
                        labels_all.append(y.detach().cpu().numpy())
                logits = np.concatenate(logits_all, axis=0)
                labels = np.concatenate(labels_all, axis=0)
                probs = _softmax_np(logits)
                st.json({"Accuracy": float((probs.argmax(axis=1) == labels).mean()),
                         "Avg CE Loss": float(np.mean(losses))})

elif mode == "Analysis":
    st.subheader("A-Flip vs. confidence/entropy vs. error (from CSVs)")

    t1 = _load_csv_anywhere(ckpt_root, "table1_performance.csv")
    t2 = _load_csv_anywhere(ckpt_root, "table2_ra_analysis.csv")
    t3 = _load_csv_anywhere(ckpt_root, "table3_model_integration.csv")

    tabs = st.tabs(["Performance", "RA analysis", "Model integration"])

    with tabs[0]:
        if t1 is not None:
            st.dataframe(t1, use_container_width=True)
            # Optional quick cards if the columns exist
            cards = {}
            for k in ["val_accuracy", "test_accuracy", "avg_loss", "num_params"]:
                if k in t1.columns and not t1.empty:
                    v = t1[k].iloc[-1]
                    cards[k.replace("_", " ").title()] = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
            if cards:
                cols = st.columns(len(cards))
                for (k, v), c in zip(cards.items(), cols):
                    c.metric(k, v)
        else:
            st.info("table1_performance.csv not found under the checkpoints root.")

    with tabs[1]:
        if t2 is not None:
            st.dataframe(t2, use_container_width=True)
            # Draw plots if expected columns are present
            # Common names the code will look for:
            #   a_flip, margin, entropy, error (0/1 or True/False)
            _safe_scatter(t2, "margin", "a_flip", "A-Flip vs margin")
            _safe_scatter(t2, "entropy", "a_flip", "A-Flip vs entropy")
            if "a_flip" in t2.columns:
                st.caption(f"Mean A-Flip: {float(t2['a_flip'].mean()):.4f}")
            if "error" in t2.columns:
                err = (t2["error"].astype(float).mean())
                st.caption(f"Error rate in sample: {err:.4f}")
        else:
            st.info("table2_ra_analysis.csv not found under the checkpoints root.")

    with tabs[2]:
        if t3 is not None:
            st.dataframe(t3, use_container_width=True)
        else:
            st.info("table3_model_integration.csv not found under the checkpoints root.")

else:
    st.subheader("Reproducibility")
    info = {"device": DEVICE, "python": f"{torch.__version__}", "torch_cuda": torch.version.cuda}
    st.json(info)
    st.markdown("**Export current settings**")
    cfg = {"ckpt_root": str(ckpt_root.resolve()), "task": task, "top_k": top_k}
    buff = io.BytesIO(json.dumps(cfg, indent=2).encode("utf-8"))
    st.download_button("Download config.json", data=buff, file_name="demo_config.json", mime="application/json")
