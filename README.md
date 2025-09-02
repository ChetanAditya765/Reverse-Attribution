# Reverse Attribution (RA) — Text & Vision

Reverse Attribution (RA) highlights **counter‑evidence**—the parts of an input that *suppress* the predicted class and would flip the decision if removed.

This repo includes a polished Streamlit demo, a click‑to‑mask interaction for **text & vision**, Captum baselines, and scripts to train/evaluate and reproduce RA analysis figures.

Youtube Demo : https://youtu.be/vf-oaVzn1iQ
---

## ✨ What’s inside

* **Unified demo**: Text (BERT on IMDB, RoBERTa on Yelp) & Vision (ResNet56 on CIFAR‑10, tiny CNN fallback).
* **Reverse Attribution** with runner‑up contrast and **A‑Flip** score.
* **Click‑to‑mask** tokens/patches → instant Δprob and A‑Flip recompute.
* **Baselines**: Captum LayerIntegratedGradients / IntegratedGradients.
* **Evaluate**: Saved metrics panel + quick compute (accuracy / CE loss).
* **Reproduction**: scripts to aggregate results and render figures.

---

## 🗂 Repo layout (key parts)

```
Reverse-Attribution/
├── app.py                         # Streamlit app
├── ra/                            # RA implementation & visualizer glue
├── models/                        # BERT/RoBERTa wrappers, ResNet56 or tiny CNN
├── scripts/                       # train/eval driver(s)
├── reproduce_results.py           # optional analysis aggregator
├── visualizer.py                  # figure renderer for aggregated results
├── checkpoints/                   # put your local weights here (not tracked)
│   ├── bert_imdb/best_model.pt
│   ├── roberta_yelp/best_model.pt
│   └── resnet56_cifar10/best_model.pt
├── reproduction_results/          # optional CSV/JSON output (tables/plots)
│   ├── table1_performance.csv
│   ├── table2_ra_analysis.csv
│   └── table3_model_integration.csv
├── requirements.txt
└── RA_NCNTAIA2025.pdf             # paper (preprint)
```

> **Note:** Weights are large—keep them out of git. See **Checkpoints** below.

---

## 🔧 Environment

<details>
<summary><strong>Windows (Anaconda Prompt / conda)</strong></summary>

```bat
cd C:\Users\cheta\OneDrive\Documents\GitHub\Reverse-Attribution

conda create -n ra-env python=3.9 -y
conda activate ra-env

:: if you have a setup.py/pyproject this enables editable mode
pip install -e .

:: pinned torch/torchtext combo known to work well
pip install torch==2.1.0 torchtext==0.16.0

:: huggingface + streaming datasets + fsspec for local/remote files
pip install -U "datasets>=2.19.1" "huggingface_hub>=0.21.2" "fsspec>=2023.12.0"

pip install ipython
pip install -r requirements.txt  :: (if present; safe to re-run)

:: Optional (helps avoid tokenizer warnings):
setx TOKENIZERS_PARALLELISM False
```

</details>

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
# (optional) conda environment
conda create -n ra-env python=3.9 -y
conda activate ra-env

# editable install if you have setup.py/pyproject
pip install -e .

# pinned torch/torchtext combo known to work well
pip install torch==2.1.0 torchtext==0.16.0

# huggingface + streaming datasets + fsspec for local/remote files
pip install -U "datasets>=2.19.1" "huggingface_hub>=0.21.2" "fsspec>=2023.12.0"

pip install ipython
pip install -r requirements.txt  # (if present; safe to re-run)

# Optional (helps avoid tokenizer warnings)
export TOKENIZERS_PARALLELISM=False
```

> Paths use `/` on Unix; drop `setx` and use `export` for env vars.

</details>

---

## 💾 Checkpoints (local only)

Create folders and drop your fine‑tuned weights as `best_model.pt`:

Hugging Space Repo : https://huggingface.co/KillBill765/Reverse-Attribution
```
checkpoints/
├── bert_imdb/best_model.pt
├── roberta_yelp/best_model.pt
└── resnet56_cifar10/best_model.pt
```

* If a weight file is missing, **text models** fall back to HF pretrained; **vision** falls back to a small CNN so the demo still runs.
* **Do not push** `.pt` files to GitHub. Use local paths, cloud storage, or HF Hub for sharing.

---

## 🏃 Training & Evaluation (scripts)

From the project root with `ra-env` activated:

```bat
:: Text model training (e.g., BERT on IMDB)
python scripts/script.py --stage train --model_type text

:: Vision model training (e.g., ResNet56/TinyCNN on CIFAR-10)
python scripts/script.py --stage train --model_type vision

:: Evaluation (produces training_history.json, CSVs, and/or JSON summary)
python scripts/script.py --stage eval
```

**Conventions (used by the app):**

Each run stores `training_history.json` under:

* `checkpoints/bert_imdb/`, `checkpoints/roberta_yelp/`, or `checkpoints/resnet56_cifar10/`.

Optional “comprehensive” summary:

* `checkpoints/<subdir>/comprehensive_evaluation_results.json`, or
* `<checkpoints_root>/comprehensive_evaluation_results.json`.

**Example JSON snippet** read by the app (flexible; only keys found are shown):

```json
{
  "standard_metrics": { "avg_loss": 0.2274, "num_samples": 25000 },
  "ra_analysis": { "summary": { "avg_a_flip": 96.67 } }
}
```

---

## 🎛 Streamlit Demo

Run the app:

```bash
streamlit run app.py
```

In the sidebar:

* Set **Checkpoints root** to your local `checkpoints` absolute path.
* Choose **Task** and **Mode**.

### Modes

**Live Demo**

* **Text**: paste a review → Predict + Explain → RA token table/heatmap.

  * Click to mask tokens (checkboxes) → **Apply mask** → see Δprob & A‑Flip (before→after).
* **Vision**: upload a 32×32 image → Predict + Explain → RA heatmap overlay.

  * Click to mask 8×8 patches → Δprob & A‑Flip recompute.

**RA vs Baselines**

* Side‑by‑side RA vs Captum LIG/IG token/patch rankings.

**Evaluate**

* **Saved Metrics (no compute)**: reads `training_history.json` and optional `comprehensive_evaluation_results.json`, shows metric cards and raw JSON.
* **Quick Evaluation (compute)**: accuracy & average cross‑entropy on IMDB/Yelp or CIFAR‑10.

**Reproducibility**

* Device/PyTorch info and a **Download `config.json`** button.

---

## 📜 Reproduce aggregated results & figures

```bash
python reproduce_results.py
python visualizer.py --input comprehensive_evaluation_results.json --outdir figs/
```

**Expected outputs:**

* `reproduction_results/` (CSVs like `table1_performance.csv`, `table2_ra_analysis.csv`, `table3_model_integration.csv`)
* `figs/` with plots exported by `visualizer.py`.

---

## 📁 Where to put CSVs (if you already have them)

If you have pre‑computed tables, place them here so the app/scripts can pick them up:

```
reproduction_results/
├── table1_performance.csv
├── table2_ra_analysis.csv
└── table3_model_integration.csv
```

---

## 🧪 Tips & Troubleshooting

* **Hugging Face downloads:** the first run may download base models/tokenizers; keep internet access on.
* **CUDA:** if you don’t have a matching CUDA build, run on CPU (it works—just slower).
* **Tokenizer missing:** loaders in `app.py` attach `model.tokenizer`; if you customize models, ensure a tokenizer is set.
* **Captum errors:** `pip install captum`.
* **Large weights:** don’t commit them; consider Git LFS/HF Hub if you must share.


---

## 📬 Contact

* **Author:** Chetan Aditya
* **Email** chetan.lakka@gmail.com

