"""
Creates an empty placeholder parquet file with the correct schema, so that
`feast apply` can validate the FeatureView's batch source before any real
data has been pushed to it. Run this once from the project root, before
running `feast apply`.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

df = pd.DataFrame({
    "user_id": pd.Series(["dummy_user"], dtype="str"),
    "event_timestamp": pd.Series([pd.Timestamp.now()], dtype="datetime64[ns]"),
    "txn_count_5min": pd.Series([0], dtype="int64"),
    "avg_amount_5min": pd.Series([0.0], dtype="float32"),
    "amount_ratio": pd.Series([0.0], dtype="float32"),
    "velocity_flag": pd.Series([False], dtype="bool"),
    "amount_spike_flag": pd.Series([False], dtype="bool"),
})

out_path = os.path.join(DATA_DIR, "user_behavior_features.parquet")
df.to_parquet(out_path)
print(f"Created placeholder parquet at: {out_path}")