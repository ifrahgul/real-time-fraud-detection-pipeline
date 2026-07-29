"""
Step 3b: Kafka Consumer — Real-Time Fraud Scoring
====================================================
Yeh script Kafka topic 'transactions' se continuously transactions
utha kar humari FastAPI '/predict' endpoint ko call karti hai, aur
result ke hisaab se decision leti hai (block/review/allow).

Extended: har prediction ab feedback.py ke through log hoti hai
(transaction_id, features, score, decision) — taake baad mein
/feedback endpoint se ground-truth label attach ki ja sake aur
retraining dataset banaya ja sake. user_id aur model latency bhi
features ke andar log hoti hain taake live dashboard unhe dikha sake.

Chalane se pehle:
    pip install kafka-python requests

Chalane ka tareeqa (Kafka aur FastAPI dono chalne chahiye):
    python kafka_consumer.py
"""

import os
import json
import time
import requests
from kafka import KafkaConsumer
from feature_store import FeatureStore
from feedback import FeedbackStore

# Docker ke andar in dono ko environment variables se override kiya jata hai
# (docker-compose.yml dekhein) — local run mein defaults (localhost) use hote hain.
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
TOPIC = "transactions"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest",   # sirf naye messages padhega, purane skip
    enable_auto_commit=True,
    group_id="fraud-detection-consumer-group",
)

feature_store = FeatureStore()
feedback_store = FeedbackStore()


def score_transaction(txn):
    """
    Transaction ko API ko bhejta hai aur fraud score wapas laata hai.
    transaction_id aur user_id ko API ko nahi bhejta (model in features
    ko nahi jaanta, yeh sirf humare rule-engine ke liye hain).
    """
    payload = {k: v for k, v in txn.items() if k not in ("transaction_id", "user_id")}

    try:
        response = requests.post(API_URL, json=payload, timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def hybrid_decision(ml_result, behavior_features):
    """
    ML score + behavioral rules ko combine karta hai (Step 6 ka
    hybrid decision engine architecture).

    Rules:
    - ML score high (>=0.9)                         -> BLOCK
    - ML score medium (>=0.6) ya behavior suspicious -> REVIEW
    - warna                                          -> ALLOW
    """
    ml_score = ml_result["fraud_probability"]
    rule_flag = behavior_features["velocity_flag"] or behavior_features["amount_spike_flag"]

    if ml_score >= 0.9:
        decision = "block"
        reason = "ML model ne high confidence fraud detect kiya"
    elif ml_score >= 0.6:
        decision = "review"
        reason = "ML model ne medium confidence fraud detect kiya"
    elif rule_flag and ml_score >= 0.3:
        decision = "review"
        reason = "Behavior suspicious hai (velocity/amount spike) + ML score borderline"
    else:
        decision = "allow"
        reason = "Normal transaction"

    return decision, reason


def run_consumer():
    print(f"[CONSUMER] Kafka broker: {KAFKA_BROKER}, Topic: {TOPIC}")
    print(f"[CONSUMER] API endpoint: {API_URL}")
    print(f"[CONSUMER] Feature store (Redis) connected: {feature_store.health_check()}")
    print("[CONSUMER] Transactions sunna shuru... (Ctrl+C se rokein)\n")

    blocked_count = 0
    review_count = 0
    allowed_count = 0

    try:
        for message in consumer:
            txn = message.value
            user_id = txn.get("user_id", "unknown_user")
            transaction_id = txn.get("transaction_id", f"unknown_{int(time.time()*1000)}")
            amount = txn.get("Amount", 0)

            start = time.perf_counter()

            # 1. ML model se score lein
            ml_result = score_transaction(txn)
            if "error" in ml_result:
                print(f"[ERROR] txn_id={transaction_id} -> {ml_result['error']}")
                continue

            # 2. Behavioral features nikaalein (is transaction se pehle ka history)
            behavior = feature_store.get_features(user_id, amount)

            # 3. Hybrid decision lein (ML + Rules)
            decision, reason = hybrid_decision(ml_result, behavior)

            # 4. Ab is transaction ko history mein record karein (agli baar ke liye)
            feature_store.record_transaction(user_id, amount)

            # 5. Prediction ko feedback store mein log karein — taake baad mein
            #    ground-truth label attach ki ja sake aur retrain dataset bane.
            #    user_id aur model ka apna latency_ms bhi features ke andar rakhte
            #    hain (dashboard mein dikhane ke liye) — model ko yeh nahi bheje jate,
            #    sirf logging ke baad add hote hain.
            log_features = {k: v for k, v in txn.items() if k != "transaction_id"}
            log_features.update(behavior)
            log_features["_latency_ms"] = ml_result.get("latency_ms")
            feedback_store.log_prediction(
                transaction_id=transaction_id,
                features=log_features,
                fraud_score=ml_result["fraud_probability"],
                decision=decision,
            )

            total_latency = (time.perf_counter() - start) * 1000

            if decision == "block":
                blocked_count += 1
                icon = "🚫"
            elif decision == "review":
                review_count += 1
                icon = "⚠️ "
            else:
                allowed_count += 1
                icon = "✅"

            print(
                f"{icon} txn_id={transaction_id} user={user_id} "
                f"amount={amount} ml_score={ml_result['fraud_probability']} "
                f"txn_count_5min={behavior['txn_count_5min']} "
                f"amount_ratio={behavior['amount_ratio']} "
                f"decision={decision.upper()} ({reason}) "
                f"[latency: {total_latency:.1f}ms]"
            )

    except KeyboardInterrupt:
        print("\n[CONSUMER] Rok diya gaya.")
        print(f"Summary -> Blocked: {blocked_count}, Review: {review_count}, Allowed: {allowed_count}")
        consumer.close()


if __name__ == "__main__":
    run_consumer()