"""
Step 3a: Kafka Producer — Transaction Stream Simulator
=========================================================
Yeh script continuously fake transactions generate kar ke Kafka topic
'transactions' mein bhejti hai — jaise real e-commerce site karti hai.

Chalane se pehle:
    pip install kafka-python

Chalane ka tareeqa (docker compose already 'up' hona chahiye):
    python kafka_producer.py
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
TOPIC = "transactions"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

# ---------------------------------------------------------
# Real dataset se asal fraud/normal rows load karte hain
# (agar mile), taake demo mein real fraud sahi score ho.
# ---------------------------------------------------------
_real_fraud_rows = None
_real_normal_rows = None

if os.path.exists(DATA_PATH):
    print(f"[PRODUCER] Real dataset mil gaya, real transactions use hongi: {DATA_PATH}")
    _df = pd.read_csv(DATA_PATH)
    _real_fraud_rows = _df[_df["Class"] == 1].drop(columns=["Class"]).to_dict("records")
    _real_normal_rows = _df[_df["Class"] == 0].drop(columns=["Class"]).to_dict("records")
else:
    print("[PRODUCER] creditcard.csv nahi mila -> synthetic random data use hogi "
          "(fraud jaisi transactions ka score kam ho sakta hai, kyunke random noise "
          "asal fraud pattern se match nahi karta).")


def generate_transaction(force_fraud=False):
    """
    Ek transaction banata hai. Agar real dataset available hai, to
    asal fraud/normal row utha kar bhejta hai (behtar demo ke liye).
    Warna synthetic (random) data generate karta hai.
    """
    if force_fraud and _real_fraud_rows:
        base = dict(random.choice(_real_fraud_rows))
    elif not force_fraud and _real_normal_rows:
        base = dict(random.choice(_real_normal_rows))
    else:
        rng = np.random.default_rng()
        if force_fraud:
            v_values = rng.normal(loc=-2.0, scale=1.5, size=28)
            amount = round(abs(rng.normal(500, 300)), 2)
        else:
            v_values = rng.normal(loc=0.0, scale=1.0, size=28)
            amount = round(abs(rng.normal(50, 60)), 2)
        base = {"Time": round(time.time() % 172800, 2), "Amount": amount}
        for i in range(1, 29):
            base[f"V{i}"] = round(float(v_values[i - 1]), 6)

    transaction = {
        "transaction_id": f"txn_{int(time.time() * 1000)}_{random.randint(1000,9999)}",
        "user_id": f"user_{random.randint(1, 8)}",  # 8 simulated users taake behavior patterns bane
    }
    transaction.update(base)
    return transaction


def run_producer(interval_seconds=2, fraud_probability=0.1):
    """
    Har `interval_seconds` mein ek transaction Kafka ko bhejta hai.
    `fraud_probability` chance hai ke woh transaction suspicious ho.
    """
    print(f"[PRODUCER] Kafka broker: {KAFKA_BROKER}, Topic: {TOPIC}")
    print("[PRODUCER] Transaction stream shuru ho rahi hai... (Ctrl+C se rokein)\n")

    count = 0
    try:
        while True:
            is_fraud_like = random.random() < fraud_probability
            txn = generate_transaction(force_fraud=is_fraud_like)

            producer.send(TOPIC, value=txn)
            producer.flush()

            count += 1
            tag = "[SUSPICIOUS PATTERN]" if is_fraud_like else "[NORMAL]"
            print(f"{count}. Sent {tag} txn_id={txn['transaction_id']} amount={txn['Amount']}")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n[PRODUCER] Rok diya gaya. Total transactions sent:", count)
        producer.close()


if __name__ == "__main__":
    # interval_seconds kam karein taake zyada tez stream simulate ho
    run_producer(interval_seconds=2, fraud_probability=0.15)
