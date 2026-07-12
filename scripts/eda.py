"""
eda.py
Exploratory analysis: what process conditions correlate with QC failure?
Saves charts to /charts for use in the final report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

df = pd.read_csv(DATA_DIR / "batch_data.csv")

numeric_cols = ["mixing_time_min", "temperature_C", "humidity_pct",
                 "moisture_content_pct", "raw_material_lot_age_days"]

# 1. Correlation-style comparison: mean of each parameter by outcome
summary = df.groupby("qc_result")[numeric_cols].mean().round(2)
print("Average process parameters by QC outcome:\n", summary, "\n")

# 2. Boxplots: numeric params vs outcome
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x="qc_result", y=col, ax=axes[i], hue="qc_result",
                palette={"Pass": "#4C9A6A", "Fail": "#C0504D"}, legend=False)
    axes[i].set_title(col)
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "param_boxplots.png", dpi=150)
plt.close()

# 3. Fail rate by categorical fields
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, ["operator_shift", "machine_id"]):
    rates = df.groupby(col)["qc_result"].apply(lambda s: (s == "Fail").mean() * 100)
    rates = rates.sort_values(ascending=False)
    sns.barplot(x=rates.index, y=rates.values, ax=ax, color="#4472C4")
    ax.set_ylabel("Fail rate (%)")
    ax.set_title(f"Fail rate by {col}")
plt.tight_layout()
plt.savefig(CHARTS_DIR / "categorical_fail_rates.png", dpi=150)
plt.close()

print("Charts saved: param_boxplots.png, categorical_fail_rates.png")
