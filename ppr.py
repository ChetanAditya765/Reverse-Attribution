#!/usr/bin/env python3
import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
EVAL_MD_PATH  = Path("evaluation_report.md")
JSON_PATH     = Path("comprehensive_evaluation_results.json")
OUTPUT_DIR    = Path("output")

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# --- 1. Parse integration status from evaluation_report.md ---
status = {}
pattern = re.compile(r"^(✅|❌)\s+(.+)$")
for line in EVAL_MD_PATH.read_text().splitlines():
    m = pattern.match(line.strip())
    if m:
        status[m.group(2).strip()] = (m.group(1) == "✅")
df_int = pd.DataFrame({
    "Model": list(status.keys()),
    "Integrated": [int(v) for v in status.values()]
})

# --- 2. Load JSON results ---
results = json.loads(JSON_PATH.read_text())

# Build performance and RA‐analysis tables
perf_rows, flip_rows = [], []
for model, info in results.items():
    std = info.get("standard_metrics", {})
    perf_rows.append({
        "Model": model,
        "Accuracy": std.get("accuracy"),
        "ECE": std.get("ece"),
    })
    for sample in info.get("ra_analysis", {}).get("detailed_results", []):
        flip_rows.append({
            "Model": model,
            "A-Flip": sample.get("a_flip")
        })

df_perf  = pd.DataFrame(perf_rows)
df_flips = pd.DataFrame(flip_rows)

# --- 3. Plot and save charts ---

# 3.1 Integration Status
plt.figure()
plt.bar(df_int["Model"], df_int["Integrated"])
plt.title("Model Integration Status")
plt.xlabel("Model")
plt.ylabel("Integrated (1 = Yes)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"integration_status.png")
plt.close()

# 3.2 Model Accuracy Comparison
plt.figure()
plt.bar(df_perf["Model"], df_perf["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"accuracy_comparison.png")
plt.close()

# 3.3 ECE vs Accuracy Scatter
plt.figure()
plt.scatter(df_perf["ECE"], df_perf["Accuracy"])
plt.title("ECE vs Accuracy")
plt.xlabel("ECE")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"ece_vs_accuracy.png")
plt.close()

# 3.4 Histogram of A-Flip Values
plt.figure()
plt.hist(df_flips["A-Flip"], bins=20)
plt.title("Histogram of A-Flip Values")
plt.xlabel("A-Flip")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"aflip_histogram.png")
plt.close()

# 3.5 Average A-Flip per Model
df_aflip_mean = df_flips.groupby("Model")["A-Flip"].mean().reset_index()
plt.figure()
plt.bar(df_aflip_mean["Model"], df_aflip_mean["A-Flip"])
plt.title("Average A-Flip per Model")
plt.xlabel("Model")
plt.ylabel("Average A-Flip")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"average_aflip.png")
plt.close()

print(f"Charts saved to '{OUTPUT_DIR}/' directory:")
print(" - integration_status.png")
print(" - accuracy_comparison.png")
print(" - ece_vs_accuracy.png")
print(" - aflip_histogram.png")
print(" - average_aflip.png")
