# app.py
from __future__ import annotations
import json
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

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

# -----------------------------
# Lazy repo-dependent imports
# -----------------------------
ReverseAttribution = None
ExplanationVisualizer = None
ExplainerHub = None
get_bert_model = None
get_resnet56_model = None

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

# -----------------------------
# Utility helpers
# -----------------------------
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

# -----------------------------
# Text quick eval helper (Accuracy + Avg CE Loss) — moved ABOVE the UI flow
# -----------------------------
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

# -----------------------------
# Text models (BERT / RoBERTa)
# -----------------------------
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

# -----------------------------
# Vision model (ResNet56 CIFAR-10 or fallback)
# -----------------------------
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

# -----------------------------
# Baselines (Captum fallback)
# -----------------------------
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

# -----------------------------
# RA helpers (optional)
# -----------------------------
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

# -----------------------------
# Presentation helpers
# -----------------------------
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
        if k in base:
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
        if k in base:
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
    st.image(heat, caption="Attribution heatmap", use_container_width=True)

def metric_cards(items: Dict[str, Any]):
    if not items:
        return
    keys = list(items.keys())
    cols = st.columns(max(1, min(4, len(keys))))
    for i, k in enumerate(keys):
        with cols[i % len(cols)]:
            st.metric(k, f"{items[k]}")

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("Controls")
    ckpt_root = Path(st.text_input("Checkpoints root", value=str((ROOT / "checkpoints").resolve())))
    task = st.selectbox("Task", ["Text: BERT (IMDB)", "Text: RoBERTa (Yelp)", "Vision: ResNet56 (CIFAR-10)"])
    mode = st.radio("Mode", ["Overview", "Live Demo", "RA vs Baselines", "Evaluate", "Reproducibility"], index=0)
    top_k = st.slider("Top-K tokens", 3, 30, 15)
    show_interactive = st.checkbox("Interactive visuals (if available)", True)
    st.caption(f"Device: **{DEVICE.upper()}**")

st.title("Reverse Attribution")

# -----------------------------
# Overview
# -----------------------------
if mode == "Overview":
    st.markdown(
        """
**Goal.** Show how Reverse Attribution (RA) highlights *counter-evidence* that flips predictions, alongside standard baselines.

**What you can do here:**
- Run **Live Demo** on text or CIFAR-10.
- Compare **RA vs Captum/LIG**.
- Run a mini **Evaluate** with Accuracy / Avg. Loss (no ECE/Brier).
- Load your trained **checkpoints** directly.
"""
    )

# -----------------------------
# Live Demo
# -----------------------------
elif mode == "Live Demo":
    if task.startswith("Text: BERT"):
        model = load_bert_imdb(ckpt_root)
        text = st.text_area("Input", "This movie was fantastic with brilliant acting!", height=120)
        if st.button("Predict + Explain"):
            ids, mask = encode_text(model, text)
            add = (mask,) if mask is not None else None
            with torch.no_grad():
                logits = forward_logits_text(model, ids, mask)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))
            st.success(f"Pred: {pred} • Prob={probs[pred]:.2f}")
            _import_ra_deps()
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
                st.warning(f"RA unavailable: {e}")
                base = run_baseline_captum(model, "bert_sentiment", {"input_ids": ids, "attention_mask": mask}, target=pred, add_args=add)
                show_text_attrib_table(model, ids, base, tokenizer=getattr(model, "tokenizer", None), attn_mask=mask, top_k=top_k)

    elif task.startswith("Text: RoBERTa"):
        model = load_roberta_yelp(ckpt_root)
        text = st.text_area("Input", "The service was quick and the food was amazing!", height=120)
        if st.button("Predict + Explain"):
            ids, mask = encode_text(model, text)
            add = (mask,) if mask is not None else None
            with torch.no_grad():
                logits = forward_logits_text(model, ids, mask)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))
            st.success(f"Pred: {pred} • Prob={probs[pred]:.2f}")
            _import_ra_deps()
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
                st.warning(f"RA unavailable: {e}")
                base = run_baseline_captum(model, "text", {"input_ids": ids, "attention_mask": mask}, target=pred, add_args=add)
                show_text_attrib_table(model, ids, base, tokenizer=getattr(model, "tokenizer", None), attn_mask=mask, top_k=top_k)

    else:
        model = load_resnet56_cifar10(ckpt_root)
        img = st.file_uploader("Upload a 32×32 RGB image", type=["png", "jpg", "jpeg"])
        if img is not None and st.button("Predict + Explain"):
            x = img32_from_uploader(img)
            with torch.no_grad():
                logits = model(x)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            pred = int(np.argmax(probs))
            st.success(f"Pred: {pred} • Prob={probs[pred]:.2f}")
            _import_ra_deps()
            try:
                ra_out = run_ra(model, x, y_true=pred, add_args=None)
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
                runner = ra_out.get("runner_up") or ra_out.get("runner_up_idx") or ra_out.get("runner_up_class")
                if a_flip is not None or runner is not None:
                    st.caption(f"A-Flip: {float(a_flip):.4f} • Pred={pred}" + (f" • Runner-up={int(runner)}" if runner is not None else ""))
            except Exception as e:
                st.warning(f"RA unavailable: {e}")
                base = run_baseline_captum(model, "resnet_cifar", x, target=pred)
                show_image_heatmap(base)

# -----------------------------
# RA vs Baselines
# -----------------------------
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

# -----------------------------
# Evaluate (no ECE/Brier anywhere)
# -----------------------------
elif mode == "Evaluate":
    st.subheader("Evaluation")

    eval_mode = st.radio(
        "Choose mode",
        ["Saved Metrics (no compute)", "Quick Evaluation (compute)"],
        index=0, horizontal=True
    )

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
            # 1) Show the readable cards first
            show = {}
            if "best/val_accuracy" in merged: show["Best Val Acc"] = f"{merged['best/val_accuracy']:.4f}"
            if "best/val_loss" in merged:     show["Best Val Loss"] = f"{merged['best/val_loss']:.4f}"
            if "epochs" in merged:            show["Epochs"] = merged["epochs"]
            if "avg_loss" in merged:          show["Avg Loss"] = f"{merged['avg_loss']:.4f}"
            if "num_samples" in merged:       show["Samples"] = merged["num_samples"]
            if "a_flip" in merged:            show["A-Flip"] = f"{merged['a_flip']:.2f}"
            metric_cards(show)

            # 2) Then show the raw JSON summary
            st.divider()  # optional
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

# -----------------------------
# Reproducibility / Downloads
# -----------------------------
else:
    st.subheader("Reproducibility")
    info = {"device": DEVICE, "python": f"{torch.__version__}", "torch_cuda": torch.version.cuda}
    st.json(info)
    st.markdown("**Export current settings**")
    cfg = {"ckpt_root": str(ckpt_root.resolve()), "task": task, "top_k": top_k}
    buff = io.BytesIO(json.dumps(cfg, indent=2).encode("utf-8"))
    st.download_button("Download config.json", data=buff, file_name="demo_config.json", mime="application/json")
