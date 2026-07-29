"""
Drift Demo — sends a burst of transactions with a deliberately shifted
amount distribution, so /health/drift visibly flips to "significant_shift".

This does NOT touch kafka_producer.py — it's a separate, one-off script
for demoing the drift detector. Run it, let the consumer process the
burst, then hit GET /health/drift.

Usage (from src/ folder, with Kafka + consumer + API already running):
    python simulate_drift.py
"""

import json
import time
import random
import numpy as np
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)


def generate_shifted_transaction():
    """
    Amount and Time are pushed far outside the training distribution
    (training Amount averages ~88, Time spans 0-172800 seconds).
    V-features are also shifted so the drift shows up clearly.
    """
    rng = np.random.default_rng()
    amount = round(abs(rng.normal(5000, 1500)), 2)   # ~50-100x normal amounts
    txn_time = round(rng.uniform(200000, 300000), 2)  # outside the training window

    v_values = rng.normal(loc=4.0, scale=2.0, size=28)  # far from training's ~0 mean

    txn = {
        "transaction_id": f"drift_{int(time.time() * 1000)}_{random.randint(1000,9999)}",
        "user_id": f"user_{random.randint(1, 8)}",
        "Time": txn_time,
        "Amount": amount,
    }
    for i in range(1, 29):
        txn[f"V{i}"] = round(float(v_values[i - 1]), 6)

    return txn


def run(n=30, interval_seconds=0.3):
    print(f"[DRIFT DEMO] Sending {n} distribution-shifted transactions to Kafka...")
    for i in range(1, n + 1):
        txn = generate_shifted_transaction()
        producer.send(TOPIC, value=txn)
        producer.flush()
        print(f"{i}. Sent shifted txn_id={txn['transaction_id']} amount={txn['Amount']} time={txn['Time']}")
        time.sleep(interval_seconds)
    print("\n[DRIFT DEMO] Done. Now hit GET /health/drift in Swagger — it should show significant_shift.")


if __name__ == "__main__":
    run()