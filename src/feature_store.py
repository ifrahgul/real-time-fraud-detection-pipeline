"""
Step 5: Real-Time Feature Store — Feast Edition
================================================
Drop-in replacement for the original Redis-only feature_store.py.
Public interface is unchanged (record_transaction, get_features, health_check)
so kafka_consumer.py needs ZERO changes to keep working with this file.

What changed under the hood:
- Raw transaction events are still kept in a Redis sorted set (same as
  before), used purely to compute the rolling 5-minute window stats.
- The *derived* behavioral features are now pushed into Feast via
  store.push(), writing to both the online store (Redis, instant serving)
  and an offline parquet log (point-in-time-correct data for retraining).
- get_features() reads back through Feast's online API, so serving
  reflects what Feast actually served, not just an in-memory dict.

One-time setup (run once, from the feature_repo/ folder):
    cd feature_repo && feast apply
"""

import os
import time
import json
import pandas as pd
import redis
from feast import FeatureStore as FeastFeatureStore
from feast.data_source import PushMode

REDIS_HOST = "localhost"
REDIS_PORT = 6379

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FEAST_REPO_PATH = os.path.join(PROJECT_ROOT, "feature_repo")

WINDOW_SECONDS = 300
VELOCITY_THRESHOLD = 5
AMOUNT_SPIKE_MULTIPLIER = 4


class FeatureStore:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, repo_path=FEAST_REPO_PATH):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.store = FeastFeatureStore(repo_path=repo_path)

    def _key(self, user_id):
        return f"user_txns:{user_id}"

    def record_transaction(self, user_id, amount, timestamp=None):
        if timestamp is None:
            timestamp = time.time()
        key = self._key(user_id)
        member = json.dumps({"amount": amount, "ts": timestamp})
        pipe = self.client.pipeline()
        pipe.zadd(key, {member: timestamp})
        pipe.zremrangebyscore(key, 0, timestamp - WINDOW_SECONDS)
        pipe.expire(key, WINDOW_SECONDS * 2)
        pipe.execute()

    def _compute_window_stats(self, user_id, current_amount, timestamp):
        key = self._key(user_id)
        raw_entries = self.client.zrangebyscore(key, timestamp - WINDOW_SECONDS, timestamp)
        amounts = []
        for entry in raw_entries:
            try:
                amounts.append(json.loads(entry)["amount"])
            except (json.JSONDecodeError, KeyError):
                continue

        txn_count = len(amounts)
        avg_amount = sum(amounts) / txn_count if txn_count > 0 else 0.0
        amount_ratio = (current_amount / avg_amount) if avg_amount > 0 else 1.0

        return {
            "user_id": user_id,
            "event_timestamp": pd.Timestamp.now(),
            "txn_count_5min": txn_count,
            "avg_amount_5min": round(avg_amount, 2),
            "amount_ratio": round(amount_ratio, 2),
            "velocity_flag": txn_count >= VELOCITY_THRESHOLD,
            "amount_spike_flag": amount_ratio >= AMOUNT_SPIKE_MULTIPLIER,
        }

    def get_features(self, user_id, current_amount, timestamp=None):
        if timestamp is None:
            timestamp = time.time()

        feats = self._compute_window_stats(user_id, current_amount, timestamp)

        df = pd.DataFrame([feats])
        try:
            self.store.push("user_behavior_push_source", df, to=PushMode.ONLINE_AND_OFFLINE)
        except Exception as e:
            print(f"[feature_store] Feast push failed, serving computed value directly: {e}")

        try:
            result = self.store.get_online_features(
                features=[
                    "user_behavior_features:txn_count_5min",
                    "user_behavior_features:avg_amount_5min",
                    "user_behavior_features:amount_ratio",
                    "user_behavior_features:velocity_flag",
                    "user_behavior_features:amount_spike_flag",
                ],
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {
                "txn_count_5min": result["txn_count_5min"][0],
                "avg_amount_5min": result["avg_amount_5min"][0],
                "amount_ratio": result["amount_ratio"][0],
                "velocity_flag": result["velocity_flag"][0],
                "amount_spike_flag": result["amount_spike_flag"][0],
            }
        except Exception as e:
            print(f"[feature_store] Feast read failed, falling back to computed value: {e}")
            return {k: feats[k] for k in (
                "txn_count_5min", "avg_amount_5min", "amount_ratio",
                "velocity_flag", "amount_spike_flag")}

    def health_check(self):
        try:
            return self.client.ping()
        except redis.exceptions.ConnectionError:
            return False


if __name__ == "__main__":
    fs = FeatureStore()
    print("Redis connected:", fs.health_check())
    user = "test_user_1"
    for amt in [50, 45, 60, 500, 480, 510]:
        fs.record_transaction(user, amt)
        print(f"Amount={amt} -> {fs.get_features(user, amt)}")
        time.sleep(0.2)