"""
generate_data.py
Generates a synthetic pharma manufacturing batch dataset with realistic,
engineered relationships between process parameters and QC outcome.

This is SYNTHETIC data (no real Strides / plant data used or needed) —
built to demonstrate the modeling pipeline end to end.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Portable paths: works from any machine, as long as the folder structure
# (data/, scripts/, models/, charts/) stays together
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)

N = 1200  # number of batches

# ---- Process parameters (each with a plausible real-world range) ----
mixing_time_min = np.random.normal(loc=45, scale=8, size=N).clip(20, 75)
temperature_C = np.random.normal(loc=28, scale=3, size=N).clip(18, 40)
humidity_pct = np.random.normal(loc=50, scale=10, size=N).clip(20, 85)
moisture_content_pct = np.random.normal(loc=3.5, scale=1.0, size=N).clip(0.5, 8)

operator_shift = np.random.choice(["Morning", "Afternoon", "Night"], size=N, p=[0.4, 0.35, 0.25])
machine_id = np.random.choice(["M1", "M2", "M3", "M4"], size=N, p=[0.3, 0.3, 0.25, 0.15])
raw_material_lot_age_days = np.random.exponential(scale=25, size=N).clip(0, 180)

# ---- Engineer a realistic failure probability from the process physics ----
# Real-world logic baked in on purpose:
#  - High humidity + long mixing time -> more moisture uptake -> more failures
#  - High temperature swings from ideal (~27C) -> more failures
#  - Older raw material lots -> slightly higher failure risk
#  - Night shift + machine M4 (oldest machine) -> small additional risk
#    (models real operational patterns: fatigue, less-maintained equipment)

risk_score = (
    0.09 * (humidity_pct - 50)
    + 0.10 * (mixing_time_min - 45)
    + 0.55 * np.abs(temperature_C - 27)
    + 0.03 * raw_material_lot_age_days
    + 0.85 * moisture_content_pct
    + np.where(operator_shift == "Night", 1.8, 0)
    + np.where(machine_id == "M4", 1.4, 0)
    - 9.0  # baseline offset so most batches pass, matching real QC pass rates
)

# small irreducible noise so it's not a trivially perfect signal (realistic QC data
# always has some unexplained variance), but the dominant pattern stays learnable
risk_score += np.random.normal(0, 1.0, size=N)

fail_prob = 1 / (1 + np.exp(-risk_score * 0.55))  # logistic squashing
qc_result = np.where(np.random.random(N) < fail_prob, "Fail", "Pass")

df = pd.DataFrame({
    "batch_id": [f"B{str(i).zfill(4)}" for i in range(1, N + 1)],
    "mixing_time_min": mixing_time_min.round(1),
    "temperature_C": temperature_C.round(1),
    "humidity_pct": humidity_pct.round(1),
    "moisture_content_pct": moisture_content_pct.round(2),
    "raw_material_lot_age_days": raw_material_lot_age_days.round(0).astype(int),
    "operator_shift": operator_shift,
    "machine_id": machine_id,
    "qc_result": qc_result,
})

df.to_csv(DATA_DIR / "batch_data.csv", index=False)

print(f"Generated {N} batch records.")
print(df["qc_result"].value_counts(normalize=True).mul(100).round(1))
print("\nSample rows:")
print(df.head())
