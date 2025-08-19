# app.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Reverse Attribution – Demo", page_icon="🧭", layout="wide")

IMPORT_WARN = None

# ---- Try repo entry points (as per README) ----
try:
    from ra import ReverseAttribution  # core RA
except Exception as e:
    ReverseAttribution = None
    IMPORT_WARN = (IMPORT_WARN or "") + f"\nReverseAttribution unavailable: {e}"
try:
    # README shows `from models import get_bert_model`
    from models import get_bert_model
except Exception as e:
    get_bert_model = None
    IMPORT_WARN = (IMPORT_WARN or "") + f"\nget_bert_model unavailable: {e}"
try:
    # Optional vision path: typical helper name in this repo
    from models import get_resnet56_model
except Exception:
    get_resnet56_model = None
try:
    # Visualizer is documented under ra.visualizer, but also exists at repo root in some branches
    try:
        from ra.visualizer import ExplanationVisualizer
    except Exception:
        from visualizer import ExplanationVisualizer  # repo-root fallback
except Exception as e:
    ExplanationVisualizer = None
    IMPORT_WARN = (IMPORT_WARN or "") + f"\nExplanationVisualizer unavailable: {e}"
try:
    # Baselines hub: Captum/SHAP/LIME wrapper
    from ra.explainer_utils import ExplainerHub  # preferred
except Exception:
    try:
        from ra.explainer_utils import ExplainerHub  # local fallback if exported differently
    except Exception as e:
        ExplainerHub = None
        IMPORT_WARN = (IMPORT_WARN or "") + f"\nExplainerHub unavailable: {e}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).parent
VIS_DIR = ROOT / "visuals"
VIS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helpers ----------

def encode_text(model, text: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not hasattr(model, "tokenizer"):
        raise RuntimeError("Model has no tokenizer attribute.")
    enc = model.tokenizer([text], return_tensors="pt", truncation=True, padding=True)
    ids = enc["input_ids"].to(DEVICE)
    mask = enc.get("attention_mask")
    if mask is not None:
        mask = mask.to(DEVICE)
    return ids, mask

def forward_logits_text(model, input_ids: torch.Tensor, attn_mask: Optional[torch.Tensor]):
    out = model(input_ids=input_ids, attention_mask=attn_mask) if attn_mask is not None else model(input_ids=input_ids)
    return out.logits if hasattr(out, "logits") else out

def load_image_32(img_file) -> torch.Tensor:
    img = Image.open(img_file).convert("RGB").resize((32, 32))
    arr = (np.array(img).astype(np.float32) / 255.0).transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

@st.cache_resource(show_spinner=False)
def load_text_model(hf_name: str = "bert-base-uncased", num_classes: int = 2):
    if get_bert_model is not None:
        m = get_bert_model(hf_name, num_classes=num_classes)
        m.to(DEVICE).eval()
        return m
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(hf_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(hf_name, num_labels=num_classes)
    mdl.tokenizer = tok
    mdl.to(DEVICE).eval()
    return mdl

@st.cache_resource(show_spinner=False)
def load_vision_model(arch: str = "resnet56", num_classes: int = 10):
    if get_resnet56_model is not None and arch == "resnet56":
        m = get_resnet56_model(num_classes=num_classes)
    else:
        try:
            from models.resnet_cifar import resnet20_cifar, resnet32_cifar
            ctor = {"resnet20": resnet20_cifar, "resnet32": resnet32_cifar}.get(arch, resnet20_cifar)
            m = ctor(num_classes=num_classes)
        except Exception as e:
            raise RuntimeError(f"Vision model loader not available: {e}")
    m.to(DEVICE).eval()
    return m

def run_ra(model, input_tensor: torch.Tensor, y_true: Optional[int] = None, add_args: Optional[tuple] = None) -> Dict[str, Any]:
    if ReverseAttribution is None:
        raise RuntimeError("ReverseAttribution not available.")
    ra = ReverseAttribution(model, device=DEVICE)
    return ra.explain(input_tensor, y_true=y_true, additional_forward_args=add_args)

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

def run_baseline_via_hub(model, model_type: str, input_tensor: Any, target: Optional[int] = None, add_args: Optional[tuple] = None) -> Dict[str, Any]:
    if ExplainerHub is None:
        raise RuntimeError("Baseline hub not available.")
    hub = None
    for kw in ({"model": model, "model_type": model_type, "device": DEVICE},
               {"model": model, "device": DEVICE},
               {"model": model}):
        try:
            hub = ExplainerHub(**kw)
            break
        except TypeError:
            continue
    if hub is None:
        raise RuntimeError("Could not construct ExplainerHub.")
    for name, params in [
        ("explain", {"input_data": input_tensor, "target_class": target, "additional_forward_args": add_args}),
        ("explain", {"input_tensor": input_tensor, "target": target, "additional_forward_args": add_args}),
        ("run",     {"input_data": input_tensor, "target_class": target, "additional_forward_args": add_args}),
        ("generate",{"input_data": input_tensor, "target_class": target, "additional_forward_args": add_args}),
        ("explain_text", {"text": input_tensor, "target": target, "additional_forward_args": add_args}),
        ("explain_image",{"image": input_tensor, "target": target, "additional_forward_args": add_args}),
    ]:
        fn = getattr(hub, name, None)
        if fn is None:
            continue
        try:
            out = fn(**{k: v for k, v in params.items() if v is not None})
            return out if isinstance(out, dict) else {"result": out}
        except TypeError:
            continue
    if hasattr(hub, "__call__"):
        out = hub(input_tensor, target=target)
        return out if isinstance(out, dict) else {"result": out}
    raise RuntimeError("No usable explain method on ExplainerHub.")

def run_baseline_captum_fallback(model, model_type: str, input_payload: Any, target: Optional[int] = None, add_args: Optional[tuple] = None) -> Dict[str, Any]:
    from captum.attr import LayerIntegratedGradients, IntegratedGradients
    model.eval()
    if model_type.startswith("bert") or model_type.startswith("text") or (isinstance(input_payload, dict) and "input_ids" in input_payload):
        if isinstance(input_payload, dict):
            input_ids = input_payload["input_ids"].to(DEVICE)
            attn = input_payload.get("attention_mask")
            attn = attn.to(DEVICE) if attn is not None else None
        else:
            input_ids = input_payload
            attn = add_args[0] if add_args else None
        emb_layer = _find_embedding_layer(model)
        if emb_layer is None:
            raise RuntimeError("Could not locate an embedding layer for text IG.")
        def fwd_ids(ids, attention_mask=None):
            out = model(input_ids=ids, attention_mask=attention_mask) if attention_mask is not None else model(input_ids=ids)
            return out.logits if hasattr(out, "logits") else out
        lig = LayerIntegratedGradients(fwd_ids, emb_layer)
        attributions = lig.attribute(inputs=input_ids, additional_forward_args=(attn,), target=target, n_steps=50)
        try:
            arr = attributions.detach().cpu().numpy()
        except Exception:
            arr = attributions
        return {"captum_lig_embeddings": arr}
    x = input_payload
    ig = IntegratedGradients(model)
    attributions = ig.attribute(inputs=x, target=target, additional_forward_args=add_args)
    try:
        arr = attributions.detach().cpu().numpy()
    except Exception:
        arr = attributions
    return {"captum_ig_image": arr}

def render_artifact(art: Any, caption_img: Optional[str] = None) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        go = None
    if isinstance(art, dict):
        p = art.get("heatmap") or art.get("image") or art.get("overlay")
        if p and Path(p).exists():
            st.image(str(p), caption=caption_img or "Explanation", use_column_width=True)
        if art.get("interactive") is not None:
            st.components.v1.html(str(art["interactive"]), height=420, scrolling=True)
        return
    if go is not None and hasattr(art, "to_plotly_json"):
        st.plotly_chart(art, use_container_width=True)
        return
    try:
        import matplotlib.figure as mplfig
        if isinstance(art, mplfig.Figure):
            st.pyplot(art)
            return
    except Exception:
        pass
    st.write(art)

def text_token_table(model, input_ids: torch.Tensor, base: Dict[str, Any], tokenizer=None, attn_mask: Optional[torch.Tensor] = None, top_k: int = 20):
    arr = None
    for k in ("captum_lig_embeddings", "captum_ig_embeddings", "captum_attribution", "result"):
        if k in base:
            arr = base[k]
            break
    if arr is None:
        st.write(base); return
    try:
        arr = arr.detach().cpu().numpy()
    except Exception:
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
    filtered: List[Tuple[str, float]] = []
    for t, s in zip(toks, scores.tolist() if isinstance(scores, np.ndarray) else list(scores)):
        if t in ("[CLS]", "[SEP]", "[PAD]", "[UNK]"):
            continue
        filtered.append((t, float(abs(s))))
    merged: List[Tuple[str, float]] = []
    for t, s in filtered:
        if t.startswith("##") and merged:
            pt, ps = merged[-1]
            merged[-1] = (pt + t[2:], ps + s)
        else:
            merged.append((t, s))
    merged.sort(key=lambda x: x[1], reverse=True)
    merged = merged[:top_k]
    st.dataframe({"token": [t for t, _ in merged], "score": [float(v) for _, v in merged]}, use_container_width=True)

def image_heatmap(base: Dict[str, Any]):
    arr = None
    for k in ("captum_ig_image", "captum_attribution", "result"):
        if k in base:
            arr = base[k]
            break
    if arr is None:
        st.write(base); return
    try:
        arr = arr.detach().cpu().numpy()
    except Exception:
        arr = np.array(arr)
    if arr.ndim == 4:
        heat = np.mean(np.abs(arr), axis=1)[0]
    elif arr.ndim == 3:
        heat = np.mean(np.abs(arr), axis=0)
    else:
        heat = np.abs(arr)
    heat = (heat - heat.min()) / (heat.ptp() + 1e-8)
    st.image(heat, use_column_width=True)

# ---------- UI ----------

with st.sidebar:
    if IMPORT_WARN:
        st.warning(IMPORT_WARN)
    st.title("Controls")
    mode = st.radio("Choose a mode", ("Quick Demo", "Baseline Comparison", "User Study", "Evaluate on Dataset", "About"), index=0)
    task = st.selectbox("Task", ["Text (BERT Sentiment)", "Vision (CIFAR-10)"])
    if task.startswith("Text"):
        hf_name = st.text_input("HF model", "bert-base-uncased")
    else:
        arch = st.selectbox("ResNet variant", ["resnet56", "resnet20", "resnet32"], index=0)
    top_m = st.slider("Top-M features for RA summary", 3, 20, 10)
    show_interactive = st.checkbox("Show interactive plots (when available)", True)
    st.caption("Device: **" + DEVICE.upper() + "**")

st.title("Reverse Attribution – Explain Uncertainty via Counter-Evidence")

if mode == "Quick Demo":
    colL, colR = st.columns([1.05, 1])
    with colL:
        st.subheader("Inference")
        if task.startswith("Text"):
            try:
                model = load_text_model(hf_name)
                text = st.text_area("Enter text", "This movie was fantastic with brilliant acting!", height=120)
                if st.button("Run Inference & RA", use_container_width=True):
                    input_ids, attn_mask = encode_text(model, text)
                    add_args = (attn_mask,) if attn_mask is not None else None
                    with torch.no_grad():
                        logits = forward_logits_text(model, input_ids, attn_mask)
                    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    st.success(f"Prediction: {pred} • Prob = {probs[pred]:.2f}")
                    st.subheader("Reverse Attribution Explanation")
                    ra_res = run_ra(model, input_ids, y_true=pred, add_args=add_args)
                    if ExplanationVisualizer is not None:
                        viz = ExplanationVisualizer(str(VIS_DIR))
                        art = viz.visualize_ra_explanation(ra_res, input_data=text, input_type="text", tokens=None, interactive=show_interactive)
                        render_artifact(art, caption_img="Token-level RA")
                    else:
                        st.json({k: str(type(v)) for k, v in ra_res.items()})
            except Exception as e:
                st.error(f"Text demo failed: {e}")
        else:
            try:
                model = load_vision_model(arch)
                img_file = st.file_uploader("Upload a CIFAR-like image (32×32)", type=["png", "jpg", "jpeg"])
                if img_file is not None and st.button("Run Inference & RA", use_container_width=True):
                    x = load_image_32(img_file)
                    with torch.no_grad():
                        logits = model(x)
                    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    st.success(f"Prediction: {pred} • Prob = {probs[pred]:.2f}")
                    st.subheader("Reverse Attribution Explanation")
                    ra_res = run_ra(model, x, y_true=pred, add_args=None)
                    if ExplanationVisualizer is not None:
                        viz = ExplanationVisualizer(str(VIS_DIR))
                        art = viz.visualize_ra_explanation(ra_res, input_data=x, input_type="image", tokens=None, interactive=show_interactive)
                        render_artifact(art, caption_img="RA overlay")
                    else:
                        st.json({k: str(type(v)) for k, v in ra_res.items()})
            except Exception as e:
                st.error(f"Vision demo failed: {e}")
    with colR:
        st.subheader("Why Reverse Attribution?")
        st.markdown(
            "- Diagnose brittle patterns (e.g., missed negations)\n"
            "- Calibrate trust with human-interpretable features\n"
            "- Compare against baseline explainers (IG/SHAP/LIME)"
        )

elif mode == "Baseline Comparison":
    st.subheader("Compare RA vs. Baselines")
    col1, col2 = st.columns(2)
    if task.startswith("Text"):
        model = load_text_model(hf_name)
        text = st.text_area("Input text", "The plot is thin but performances are superb.")
        input_ids, attn_mask = encode_text(model, text)
        add_args = (attn_mask,) if attn_mask is not None else None
        with torch.no_grad():
            logits = forward_logits_text(model, input_ids, attn_mask)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(probs))
        with col1:
            st.markdown("**Reverse Attribution**")
            try:
                ra_res = run_ra(model, input_ids, y_true=pred, add_args=add_args)
                if ExplanationVisualizer is not None:
                    viz = ExplanationVisualizer(str(VIS_DIR))
                    art = viz.visualize_ra_explanation(ra_res, input_data=text, input_type="text", tokens=None, interactive=show_interactive)
                    render_artifact(art)
            except Exception as e:
                st.error(f"RA failed: {e}")
        with col2:
            use_fallback = ExplainerHub is None
            st.markdown("**Captum/SHAP/LIME (via ExplainerHub)**" if not use_fallback else "**Captum (fallback, Integrated Gradients)**")
            try:
                if use_fallback:
                    payload = {"input_ids": input_ids}
                    if add_args: payload["attention_mask"] = add_args[0]
                    base = run_baseline_captum_fallback(model, "bert_sentiment", payload, target=pred, add_args=add_args)
                else:
                    base = run_baseline_via_hub(model, "bert_sentiment", input_ids, target=pred, add_args=add_args)
                text_token_table(model, input_ids, base, tokenizer=getattr(model, "tokenizer", None), attn_mask=attn_mask, top_k=top_m)
            except Exception as e:
                st.error(f"Baseline failed: {e}")
    else:
        model = load_vision_model(arch)
        img_file = st.file_uploader("Upload image (32×32)", type=["png", "jpg", "jpeg"])
        if img_file is not None:
            x = load_image_32(img_file)
            with torch.no_grad():
                logits = model(x)
            probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = int(np.argmax(probs))
            with col1:
                st.markdown("**Reverse Attribution**")
                try:
                    ra_res = run_ra(model, x, y_true=pred)
                    if ExplanationVisualizer is not None:
                        viz = ExplanationVisualizer(str(VIS_DIR))
                        art = viz.visualize_ra_explanation(ra_res, input_data=x, input_type="image", tokens=None, interactive=show_interactive)
                        render_artifact(art)
                except Exception as e:
                    st.error(f"RA failed: {e}")
            with col2:
                use_fallback = ExplainerHub is None
                st.markdown("**Captum/SHAP/LIME (via ExplainerHub)**" if not use_fallback else "**Captum (fallback, Integrated Gradients)**")
                try:
                    if use_fallback:
                        base = run_baseline_captum_fallback(model, "resnet_cifar", x, target=pred)
                    else:
                        base = run_baseline_via_hub(model, "resnet_cifar", x, target=pred)
                    image_heatmap(base)
                except Exception as e:
                    st.error(f"Baseline failed: {e}")

elif mode == "User Study":
    st.subheader("Trust Calibration & Debugging Time (Toy)")
    pid = st.text_input("Participant ID", value="user_" + str(int(time.time())))
    tab1, tab2 = st.tabs(["Trust Calibration", "Debugging Time"])
    with tab1:
        if task.startswith("Text"):
            try:
                model = load_text_model(hf_name)
                text = st.text_area("Sample text", "I did not enjoy the movie although the visuals were stunning.")
                input_ids, attn_mask = encode_text(model, text)
                before = st.slider("Before-explanation trust", 1, 5, 3)
                after = st.slider("After-explanation trust", 1, 5, 4)
                if st.button("Record Trust Entry"):
                    st.success(f"Recorded for {pid}: before={before}, after={after}")
            except Exception as e:
                st.error(f"Trust demo failed: {e}")
        else:
            st.info("Switch to Text task to try the trust calibration flow.")
    with tab2:
        if st.button("Start Timer"): st.session_state["dbg_start"] = time.time()
        if st.button("Stop & Record"):
            elapsed = time.time() - st.session_state.get("dbg_start", time.time())
            st.success(f"Recorded debug time: {elapsed:.1f} s (participant {pid})")

elif mode == "Evaluate on Dataset":
    st.subheader("Mini Evaluation (Accuracy / ECE / Brier)")

    def _softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            m, M = bins[i], bins[i + 1]
            idx = (conf > m) & (conf <= M)
            if np.any(idx):
                acc = (preds[idx] == labels[idx]).mean()
                gap = abs(acc - conf[idx].mean())
                ece += idx.mean() * gap
        return float(ece)

    def _brier(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
        y = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
        y[np.arange(labels.shape[0]), labels] = 1.0
        return float(np.mean((probs - y) ** 2))

    def _fallback_eval_text(hf_name: str, max_samples: int = 256, batch_size: int = 32) -> Dict[str, Any]:
        from datasets import load_dataset
        model = load_text_model(hf_name)
        tok = model.tokenizer
        ds = load_dataset("imdb", split=f"test[:{max_samples}]")
        def _collate(batch):
            enc = tok([x["text"] for x in batch], return_tensors="pt", truncation=True, padding=True)
            labels = torch.tensor([int(x["label"]) for x in batch])
            return enc, labels
        from torch.utils.data import DataLoader
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
        logits_all, labels_all = [], []
        model.eval()
        with torch.no_grad():
            for enc, y in dl:
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                out = model(**enc)
                logits = out.logits if hasattr(out, "logits") else out
                logits_all.append(logits.detach().cpu().numpy())
                labels_all.append(y.numpy())
        logits = np.concatenate(logits_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)
        probs = _softmax_np(logits)
        return {"Accuracy": float((probs.argmax(axis=1) == labels).mean()),
                "ECE": _ece(probs, labels),
                "Brier": _brier(probs, labels, probs.shape[1])}

    def _fallback_eval_vision(arch: str, max_samples: int = 512, batch_size: int = 64) -> Dict[str, Any]:
        import torchvision
        from torch.utils.data import DataLoader
        from torchvision import transforms
        model = load_vision_model(arch)
        tfm = transforms.ToTensor()
        ds = torchvision.datasets.CIFAR10(root=str(ROOT/"data"), train=False, download=True, transform=tfm)
        if max_samples < len(ds): ds = torch.utils.data.Subset(ds, range(max_samples))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        logits_all, labels_all = [], []
        model.eval()
        with torch.no_grad():
            for x, y in dl:
                x = x.to(DEVICE)
                out = model(x)
                logits_all.append(out.detach().cpu().numpy())
                labels_all.append(y.numpy())
        logits = np.concatenate(logits_all, axis=0)
        labels = np.concatenate(labels_all, axis=0)
        probs = _softmax_np(logits)
        return {"Accuracy": float((probs.argmax(axis=1) == labels).mean()),
                "ECE": _ece(probs, labels),
                "Brier": _brier(probs, labels, probs.shape[1])}

    try:
        if task.startswith("Text"):
            st.json(_fallback_eval_text(hf_name))
        else:
            st.json(_fallback_eval_vision(arch))
    except Exception as e:
        st.error(f"Evaluation failed: {e}")

else:
    st.header("About")
    st.write("This app showcases Reverse Attribution across NLP and Vision with baseline comparisons and a portable mini-eval.")
