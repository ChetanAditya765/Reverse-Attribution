# app.py
"""
Streamlit dashboard for Reverse Attribution framework.
Integrates:
- Model selection and loading (BERTSentimentClassifier, ResNetCIFAR)
- Reverse Attribution explanations (RA)
- Baseline explainers (SHAP, LIME, Captum)
- Visualization of explanations
- User study task interfaces (trust calibration, debugging time)
"""

import streamlit as st
import torch
from pathlib import Path
import time

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

# Core RA and utilities
from ra import ReverseAttribution, ModelFactory, ExplanationVisualizer
from ra.explainer_utils import ExplainerHub
from ra.user_study import new_session, UserStudyAnalyzer
from ra.dataset_utils import DatasetLoader

# Models
from models import get_bert_model, get_resnet56_model

st.set_page_config(page_title="Reverse Attribution Demo", layout="wide")

# Sidebar: choose mode
mode = st.sidebar.selectbox(
    "Choose mode",
    ["Inference & Explanation", "Baseline Comparison", "User Study"]
)

if mode == "Inference & Explanation":
    st.title("🔍 Model Inference & RA Explanation")

    # Model selection
    model_type = st.sidebar.selectbox("Model Type", ["Text (BERT)", "Vision (ResNet)"])
    if model_type == "Text (BERT)":
        model_name = st.sidebar.text_input("HF Model Name", "bert-base-uncased")
        if get_bert_model:
            model = get_bert_model(model_name, num_classes=2)
            tokenizer = model.tokenizer
        else:
            st.error("BERT model unavailable.")
            st.stop()
    else:
        arch = st.sidebar.selectbox("ResNet Architecture", ["resnet56", "resnet20", "resnet32"])
        if get_resnet56_model and arch == "resnet56":
            model = get_resnet56_model(num_classes=10)
            tokenizer = None
        else:
            # fallback for other architectures
            from models.resnet_cifar import resnet20_cifar, resnet32_cifar
            model = {"resnet20": resnet20_cifar, "resnet32": resnet32_cifar}[arch](num_classes=10)
            tokenizer = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # User input
    if model_type == "Text (BERT)":
        text = st.text_area("Enter text to explain:", "This movie was fantastic!")
        if st.button("Explain Text"):
            encoded = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded["attention_mask"].to(device)
            ra = ReverseAttribution(model, device=device)
            result = ra.explain(input_ids, y_true=1, additional_forward_args=(attn_mask,))
            viz = ExplanationVisualizer("visuals")
            artifacts = viz.visualize_ra_explanation(result, tokenizer=tokenizer)
            st.image(artifacts["heatmap"], caption="Token Heatmap")
            st.write("Interactive plot:", artifacts["interactive"])
    else:
        img_file = st.file_uploader("Upload image (32×32 RGB):", type=["png","jpg"])
        if img_file and st.button("Explain Image"):
            import numpy as np
            from PIL import Image
            img = Image.open(img_file).resize((32,32))
            img_tensor = torch.tensor(np.array(img).transpose(2,0,1)/255.0).unsqueeze(0).to(device)
            ra = ReverseAttribution(model, device=device)
            result = ra.explain(img_tensor, y_true=0)
            viz = ExplanationVisualizer("visuals")
            artifacts = viz.visualize_ra_explanation(result, input_data=img_tensor)
            st.image(artifacts["overlay"], caption="RA Overlay")

elif mode == "Baseline Comparison":
    st.title("⚖️ Baseline Explainers Comparison")
    # Similar UI: choose model, input, then compare RA vs SHAP vs LIME vs Captum
    model_type = st.sidebar.selectbox("Model Type", ["Text", "Vision"])
    # ... implement analogous to above but using ExplainerHub
    st.info("Baseline comparison UI to be implemented")

else:  # User Study
    st.title("🧪 User Study")
    study_mode = st.sidebar.selectbox("Study Type", ["Trust Calibration", "Debugging Time"])
    participant_id = st.sidebar.text_input("Participant ID", value="user_" + torch.randint(0,9999,(1,)).item().__str__())
    session = new_session(participant_id, study_name=study_mode.replace(" ","_").lower())

    loader = DatasetLoader()
    if study_mode == "Trust Calibration":
        st.header("Trust Calibration Study")
        text = st.text_area("Sample text:", "The product is great!")
        if st.button("Record Trust"):
            # dummy before/after
            before = st.slider("Before Explanation Trust:",1,5,3)
            after  = st.slider("After Explanation Trust:",1,5,4)
            session.record_trust("sample_1","ra", before, after, {"text": text})
            st.success("Recorded!")
    else:
        st.header("Debugging Time Study")
        start = st.button("Start Task")
        if start:
            st.session_state.start_time = time.time()
        if st.button("End Task"):
            elapsed = time.time() - st.session_state.get("start_time", time.time())
            success = st.checkbox("Task Solved?", value=True)
            session.record_debug_time("sample_1","ra", elapsed, success, {})
            st.success(f"Recorded {elapsed:.1f}s")

    if st.button("Save Session"):
        path = session.save()
        st.write(f"Saved to {path}")

    if st.button("Analyze Study"):
        analyzer = UserStudyAnalyzer(f"user_study_data/{study_mode.replace(' ','_').lower()}.csv")
        st.markdown("### Trust Change")
        st.write(analyzer.trust_change())
        st.markdown("### Debugging Stats")
        st.write(analyzer.debugging_stats())
