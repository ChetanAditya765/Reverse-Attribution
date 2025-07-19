# Reverse Attribution

**Reverse Attribution** is a framework for explaining model uncertainty and failures via counter-evidence analysis. It identifies features whose removal most suppresses the true-class prediction, providing an “attribution-flip” instability score (A-Flip) and counter-evidence features.

## Features

-   **Core RA algorithm** (`ra.ReverseAttribution`) for both text (BERT-based) and vision (ResNet-CIFAR) models
-   **Integrated model implementations**:
    -   `BERTSentimentClassifier` for sentiment tasks
    -   `ResNetCIFAR` (ResNet-56, etc.) for CIFAR-10/100
-   **Baseline explainers** (SHAP, LIME, Captum) via `ra.explainer_utils.ExplainerHub`
-   **Visualization utilities** (`ra.visualizer`) for heatmaps, overlays, word clouds, interactive plots
-   **Training & evaluation scripts** (`scripts/`) aligned with the JMLR paper
-   **User-study framework** (`ra.user_study`) for trust calibration & debugging-time experiments
-   **Reproduction pipeline** (`reproduce_results.py`) generating all figures, tables, and reports

## Installation

Install using pip:

```bash
pip install -e .
```

Or using Conda:

```bash
python setup_environment.py
conda activate reverse-attribution
```

## Quickstart

Here is a brief example of how to use the library.

```python
from models import get_bert_model
from ra import ReverseAttribution
from ra.visualizer import ExplanationVisualizer

# Load model and tokenizer
model = get_bert_model("bert-base-uncased", num_classes=2)

# Create an explainer instance
ra_explainer = ReverseAttribution(model)

# Explain a sample
input_ids = model.tokenizer("Amazing movie!", return_tensors="pt")["input_ids"]
# explanation = ra_explainer.explain(input_ids) # Generate explanation

# Visualize the results
# Note: The line below is based on the provided text and may be incomplete.
viz = ExplanationVisualizer("figs")
# viz.visualize_ra_explanation(...) 
```

---

See the User Guide and API Reference for detailed instructions.
