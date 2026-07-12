"""
generate_new_batches.py
Simulates a fresh set of "incoming" batches -- like a new week's production --
that haven't been through final QC yet. These are what the trained model will
score in score_batches.py.

Uses a DIFFERENT random seed than the training data, so this genuinely acts
like unseen, real-time-style data rather than a copy of the training set.
"""

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(101)  # different seed from generate_data.py -- genuinely new batches

N = 20  # a realistic-sized "incoming batch" list, e.g. one week's production

mixing_time_min = np.random.normal(loc=45, scale=8, size=N).clip(20, 75)
temperature_C = np.random.normal(loc=28, scale=3, size=N).clip(18, 40)
humidity_pct = np.random.normal(loc=50, scale=10, size=N).clip(20, 85)
moisture_content_pct = np.random.normal(loc=3.5, scale=1.0, size=N).clip(0.5, 8)
operator_shift = np.random.choice(["Morning", "Afternoon", "Night"], size=N, p=[0.4, 0.35, 0.25])
machine_id = np.random.choice(["M1", "M2", "M3", "M4"], size=N, p=[0.3, 0.3, 0.25, 0.15])
raw_material_lot_age_days = np.random.exponential(scale=25, size=N).clip(0, 180)

# Note: no qc_result column here -- these batches haven't been through
# final QC yet. That's the entire point: we're predicting it before it happens.
df = pd.DataFrame({
    "batch_id": [f"NEW{str(i).zfill(3)}" for i in range(1, N + 1)],
    "mixing_time_min": mixing_time_min.round(1),
    "temperature_C": temperature_C.round(1),
    "humidity_pct": humidity_pct.round(1),
    "moisture_content_pct": moisture_content_pct.round(2),
    "raw_material_lot_age_days": raw_material_lot_age_days.round(0).astype(int),
    "operator_shift": operator_shift,
    "machine_id": machine_id,
})

df.to_csv(DATA_DIR / "new_batches.csv", index=False)
print(f"Generated {N} new (unscored) incoming batches -> data/new_batches.csv")
print(df.head())
