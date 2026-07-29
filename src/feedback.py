"""
Feedback Loop — closes the MLOps cycle.
=========================================
Logs every prediction (features + score + decision) and lets you attach a
ground-truth label later (analyst review, chargeback, customer report).
That labeled data is what makes real retraining possible, and lets you
compute real precision/recall instead of only training-set metrics.

Uses SQLite (a single file) so both kafka_consumer.py (writer) and
api.py (reader, for /feedback and /health/metrics) can share it without
running a separate database service.
"""

import sqlite3
import json
import time
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DB_PATH = os.path.join(PROJECT_ROOT, "data", "feedback.db")


class FeedbackStore:
    def __init__(self, db_path=DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                transaction_id TEXT PRIMARY KEY,
                features TEXT NOT NULL,
                fraud_score REAL NOT NULL,
                decision TEXT NOT NULL,
                predicted_at REAL NOT NULL,
                actual_label INTEGER,
                labeled_at REAL
            )
        """)
        self.conn.commit()

    def log_prediction(self, transaction_id, features: dict, fraud_score: float, decision: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO predictions "
            "(transaction_id, features, fraud_score, decision, predicted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (transaction_id, json.dumps(features), fraud_score, decision, time.time()),
        )
        self.conn.commit()

    def record_feedback(self, transaction_id, actual_label: int):
        """actual_label: 1 = confirmed fraud, 0 = confirmed legitimate"""
        cur = self.conn.execute(
            "UPDATE predictions SET actual_label = ?, labeled_at = ? WHERE transaction_id = ?",
            (actual_label, time.time(), transaction_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def get_live_metrics(self):
        rows = self.conn.execute(
            "SELECT fraud_score, decision, actual_label FROM predictions "
            "WHERE actual_label IS NOT NULL"
        ).fetchall()
        if not rows:
            return {"status": "no_labeled_data_yet", "labeled_count": 0}

        tp = sum(1 for _, decision, label in rows if decision == "block" and label == 1)
        fp = sum(1 for _, decision, label in rows if decision == "block" and label == 0)
        fn = sum(1 for _, decision, label in rows if decision != "block" and label == 1)

        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None

        return {
            "labeled_count": len(rows),
            "precision": round(precision, 3) if precision is not None else None,
            "recall": round(recall, 3) if recall is not None else None,
        }

    def export_labeled_dataset(self, out_path=None):
        import pandas as pd
        out_path = out_path or os.path.join(PROJECT_ROOT, "data", "retrain_dataset.csv")
        rows = self.conn.execute(
            "SELECT features, actual_label FROM predictions WHERE actual_label IS NOT NULL"
        ).fetchall()
        records = [{**json.loads(f), "label": label} for f, label in rows]
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path, len(df)
