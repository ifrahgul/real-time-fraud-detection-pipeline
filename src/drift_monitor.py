"""
Drift Monitor — detects when live traffic no longer looks like training data.
==============================================================================
Population Stability Index (PSI), the metric fraud/risk teams actually use:
    PSI < 0.1   -> stable
    0.1 - 0.25  -> moderate shift, worth watching
    > 0.25      -> significant shift, retrain likely needed
"""

import json
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
BASELINE_PATH = os.path.join(PROJECT_ROOT, "data", "baseline_distributions.json")
N_BINS = 10


class DriftMonitor:
    def __init__(self, baseline_path=BASELINE_PATH):
        self.baseline_path = baseline_path
        self.baselines = self._load_baseline()

    def _load_baseline(self):
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _bin_edges(series, n_bins=N_BINS):
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(series.quantile(quantiles).values)
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    @classmethod
    def save_baseline(cls, df: pd.DataFrame, columns, path=BASELINE_PATH):
        """Call this once, right after training, on the training feature set."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        baselines = {}
        for col in columns:
            edges = cls._bin_edges(df[col])
            counts, _ = np.histogram(df[col], bins=edges)
            proportions = (counts / counts.sum()).tolist()
            baselines[col] = {"edges": edges.tolist(), "proportions": proportions}
        with open(path, "w") as f:
            json.dump(baselines, f)
        return baselines

    def _psi_for_column(self, col, live_series):
        base = self.baselines[col]
        edges = np.array(base["edges"])
        base_props = np.array(base["proportions"])

        counts, _ = np.histogram(live_series, bins=edges)
        live_props = counts / max(counts.sum(), 1)

        base_props = np.clip(base_props, 1e-4, None)
        live_props = np.clip(live_props, 1e-4, None)

        return float(np.sum((live_props - base_props) * np.log(live_props / base_props)))

    def check_drift(self, live_df: pd.DataFrame):
        if not self.baselines:
            return {"status": "no_baseline", "detail": "Run save_baseline() after training first."}

        report = {}
        for col in self.baselines:
            if col not in live_df.columns:
                continue
            psi = self._psi_for_column(col, live_df[col])
            if psi < 0.1:
                status = "stable"
            elif psi < 0.25:
                status = "moderate_shift"
            else:
                status = "significant_shift"
            report[col] = {"psi": round(psi, 4), "status": status}

        worst = max(report.values(), key=lambda r: r["psi"]) if report else None
        overall = worst["status"] if worst else "unknown"
        return {"status": overall, "features": report}
