# User Guide

This guide walks through the complete process of setting up the environment, training models, running evaluations, and using the interactive features of the Reverse Attribution framework.

---

## 1. Environment Setup

First, set up the Conda environment and install the required dependencies.

```bash
# Run the setup script to create the conda environment
python setup_environment.py

# Activate the environment
conda activate reverse-attribution

# Install the project in editable mode
pip install -e .
```

---

## 2. Training Models

You can train both text and vision models using the main script.

### Text Models

To train the IMDB BERT sentiment classification model, run:

```bash
python scripts/script.py --stage train --model_type text
```

### Vision Models

To train a ResNet-56 model on the CIFAR-10 dataset, run:

```bash
python scripts/script.py --stage train --model_type vision
```

---

## 3. Evaluation

After training, you can evaluate the models to measure their performance and generate explanations.

```bash
python scripts/script.py --stage eval
```

---

## 4. Reproduce All Results

To reproduce all figures, tables, and reports from the paper, run the dedicated reproduction script.

```bash
python reproduce_results.py --all
```

This will generate a `reproduction_results/` directory containing all artifacts, including a detailed `reproduction_report.md`.

---

## 5. Interactive Demo

An interactive Streamlit application is available for hands-on exploration.

Launch the app by running:
```bash
streamlit run app.py
```

The app provides the following functionalities:
-   **Inference & Explanation**: Select a text or image model, provide custom input, and view the generated Reverse Attribution explanations.
-   **Baseline Comparison**: Compare RA explanations against baseline methods like SHAP, LIME, and Captum.
-   **User Study**: A framework to conduct trust calibration and debugging-time experiments.

---

## 6. Visualization

The library includes utilities to create and save static and interactive visualizations of explanations. These are saved under the `figs/` directory by default.

```python
from ra.visualizer import ExplanationVisualizer

# Initialize the visualizer
viz = ExplanationVisualizer("figs")

# Example: Visualize a Reverse Attribution result
# (Assumes 'ra_results' and 'model' are already loaded)
viz.visualize_ra_explanation(ra_results, tokenizer=model.tokenizer)
```

---

## 7. User Studies

The Streamlit app provides a front-end for conducting user studies.

### Trust Calibration
1.  Navigate to **Streamlit → User Study → Trust Calibration**.
2.  Follow the on-screen instructions to record pre- and post-explanation trust responses.
3.  Analyze the collected data using the `UserStudyAnalyzer`.

### Debugging Time
1.  Navigate to **Streamlit → User Study → Debugging Time**.
2.  Use the interface to record start/end times and a success flag for debugging tasks.
3.  Analyze the collected data using the `UserStudyAnalyzer`.
