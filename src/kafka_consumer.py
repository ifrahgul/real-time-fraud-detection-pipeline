"""
Step 3b: Kafka Consumer — Real-Time Fraud Scoring
====================================================

"""

import os
import json
import time
import requests
from kafka import KafkaConsumer
from feature_store import FeatureStore
from feedback import FeedbackStore

# Inside Docker, these two are overridden via environment variables
# (see docker-compose.yml) — for local runs, the defaults (localhost) are used.
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
TOPIC = "transactions"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=KAFKA_BROKER,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="latest",   # only reads new messages, skips old ones
    enable_auto_commit=True,
    group_id="fraud-detection-consumer-group",
)

feature_store = FeatureStore()
feedback_store = FeedbackStore()


def score_transaction(txn):
    """
    Sends the transaction to the API, excluding transaction_id and user_id
    (the model doesn't know about these features, they're only for our rule engine).
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
    Combines ML score + behavioral rules (Step 6's
    hybrid decision engine architecture).

    Rules:
    - ML score high (>=0.9)                          -> BLOCK
    - ML score medium (>=0.6) or behavior suspicious  -> REVIEW
    - otherwise                                       -> ALLOW
    """
    ml_score = ml_result["fraud_probability"]
    rule_flag = behavior_features["velocity_flag"] or behavior_features["amount_spike_flag"]

    if ml_score >= 0.9:
        decision = "block"
        reason = "ML model shows high confidence score"
    elif ml_score >= 0.6:
        decision = "review"
        reason = "ML model detected medium score"
    elif rule_flag and ml_score >= 0.3:
        decision = "review"
        reason = "Behavior is suspicious (velocity/amount spike) + ML score borderline"
    else:
        decision = "allow"
        reason = "Normal transaction"

    return decision, reason


def run_consumer():
    print(f"[CONSUMER] Kafka broker: {KAFKA_BROKER}, Topic: {TOPIC}")
    print(f"[CONSUMER] API endpoint: {API_URL}")
    print(f"[CONSUMER] Feature store (Redis) connected: {feature_store.health_check()}")
    print("[CONSUMER] Starting to listen for transactions... (Ctrl+C to stop)\n")

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

            # 1. Get score from the ML model
            ml_result = score_transaction(txn)
            if "error" in ml_result:
                print(f"[ERROR] txn_id={transaction_id} -> {ml_result['error']}")
                continue

            # 2. Extract behavioral features (history prior to this transaction)
            behavior = feature_store.get_features(user_id, amount)

            # 3. Make the hybrid decision (ML + Rules)
            decision, reason = hybrid_decision(ml_result, behavior)

            # 4. Now record this transaction in history (for next time)
            feature_store.record_transaction(user_id, amount)

            # 5. Log the prediction in the feedback store — so a
            #    ground-truth label can be attached later and a retrain dataset built.
            #    We also keep user_id and the model's own latency_ms inside the features
            #    (for showing in the dashboard) — these aren't sent to the model,
            #    they're only added after logging.
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
        print("\n[CONSUMER] Stopped.")
        print(f"Summary -> Blocked: {blocked_count}, Review: {review_count}, Allowed: {allowed_count}")
        consumer.close()


if __name__ == "__main__":
    run_consumer()
