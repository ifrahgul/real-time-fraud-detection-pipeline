"""
Step 2: FastAPI Serving Endpoint
==================================
Yeh API trained model (fraud_model.pkl) ko load karti hai aur ek
/predict endpoint deti hai jahan aap transaction bhej kar turant
fraud score le sakte hain.

Extended with:
    /explain            -> SHAP-based reason for a given transaction's score
    /feedback            -> record the confirmed outcome of a past prediction
    /health/metrics       -> live precision/recall from confirmed feedback
    /health/drift         -> feature drift status (PSI) vs training baseline
    /retrain/export       -> export feedback-labeled transactions for retraining
    /live/transactions     -> recent real-time predictions (for the live dashboard)
    /live/summary          -> aggregate KPIs across all logged predictions

Chalane ka tareeqa (is file ke folder se):
    uvicorn api:app --reload --port 8000

Test karne ka tareeqa:
    http://127.0.0.1:8000/docs   (Swagger UI - browser mein khulega)
"""

import os
import pickle
import json
import time

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from drift_monitor import DriftMonitor
from explainability import Explainer
from feedback import FeedbackStore

# ---------------------------------------------------------
# PATHS (src/ folder se ek level upar models/ aur data/ folders)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MODEL_PATH = os.path.join(MODELS_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_names.json")

# Fraud score decision thresholds (Step 1 project idea ke rules se)
BLOCK_THRESHOLD = 0.9
REVIEW_THRESHOLD = 0.6

# Live dashboard ke liye kitni recent predictions "recent" gini jayein
# (avg latency waghera isi window se calculate hoti hai)
LIVE_WINDOW_SIZE = 200


# ---------------------------------------------------------
# LOAD MODEL, SCALER, FEATURE LIST (server start hote hi ek dafa)
# ---------------------------------------------------------
def load_artifacts():
    for path, name in [(MODEL_PATH, "Model"), (SCALER_PATH, "Scaler"), (FEATURES_PATH, "Feature list")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} nahi mila: {path}\n"
                f"Pehle 'python train_model.py' chalayein taake yeh files ban sakein."
            )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)

    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()

# --- new: explainability, drift monitoring, feedback store ---
explainer = Explainer(model)
drift_monitor = DriftMonitor()
feedback_store = FeedbackStore()


# ---------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------
class Transaction(BaseModel):
    """
    Ek transaction ka input format. V1-V28 Kaggle dataset ke PCA features hain.
    Real system mein yeh features aapke apne raw transaction data se
    (amount, location, device, etc.) engineer honge.
    """
    Time: float = Field(..., description="Transaction ke pehle transaction se seconds ka fasla")
    Amount: float = Field(..., description="Transaction amount")
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0


class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: str          # "block" | "review" | "allow"
    latency_ms: float


class FeedbackRequest(BaseModel):
    transaction_id: str
    actual_label: int  # 1 = confirmed fraud, 0 = confirmed legitimate


# ---------------------------------------------------------
# APP
# ---------------------------------------------------------
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Trained XGBoost model se transactions ko real-time score karti hai.",
    version="1.2.0",
)

# CORS enable karna zaroori hai taake dashboard.html (browser se, koi backend
# server ke bina) seedha is API ko fetch() kar sake.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _prepare_input(transaction: Transaction) -> pd.DataFrame:
    """Shared prep logic used by both /predict and /explain, so they never drift apart."""
    input_dict = transaction.dict()
    df = pd.DataFrame([input_dict])
    df = df[feature_names]  # training ke waqt jo order tha wahi order
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    return df


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    start_time = time.perf_counter()

    try:
        df = _prepare_input(transaction)
        fraud_proba = float(model.predict_proba(df)[0][1])

        if fraud_proba >= BLOCK_THRESHOLD:
            decision = "block"
        elif fraud_proba >= REVIEW_THRESHOLD:
            decision = "review"
        else:
            decision = "allow"

        latency_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResponse(
            fraud_probability=round(fraud_proba, 4),
            decision=decision,
            latency_ms=round(latency_ms, 3),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/explain")
def explain(transaction: Transaction):
    """Returns the fraud score plus the top SHAP features driving it."""
    try:
        df = _prepare_input(transaction)
        feature_row = df.iloc[0].to_dict()
        return explainer.explain(feature_row)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explain failed: {str(e)}")


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Call this once a human / chargeback / dispute process confirms whether
    a past transaction (logged by kafka_consumer.py) really was fraud.
    """
    updated = feedback_store.record_feedback(req.transaction_id, req.actual_label)
    if not updated:
        raise HTTPException(status_code=404, detail="transaction_id not found in feedback log")
    return {"status": "recorded"}


@app.get("/health/metrics")
def live_metrics():
    """Real precision/recall computed only from confirmed feedback, not training data."""
    return feedback_store.get_live_metrics()


@app.get("/health/drift")
def drift_status():
    """
    Drift check against the training baseline, using recently labeled/logged
    transactions. Run drift_monitor.save_baseline() once after training first.
    """
    rows = feedback_store.conn.execute(
        "SELECT features FROM predictions ORDER BY predicted_at DESC LIMIT 200"
    ).fetchall()
    if not rows:
        return {"status": "no_data_yet"}
    recent_df = pd.DataFrame([json.loads(r[0]) for r in rows])
    return drift_monitor.check_drift(recent_df)


@app.post("/retrain/export")
def export_retrain_data():
    """Pulls all feedback-labeled transactions into a CSV ready for the next training run."""
    path, count = feedback_store.export_labeled_dataset()
    return {"exported_to": path, "row_count": count}


# ---------------------------------------------------------
# LIVE DASHBOARD ENDPOINTS
# ---------------------------------------------------------
@app.get("/live/transactions")
def live_transactions(limit: int = 100):
    """
    Real-time pipeline se scored transactions wapas karta hai — jo
    kafka_consumer.py ne feedback_store mein log ki thin. Dashboard yeh
    endpoint har 2 second mein poll karta hai.
    """
    limit = max(1, min(limit, 500))
    rows = feedback_store.conn.execute(
        "SELECT transaction_id, features, fraud_score, decision, predicted_at, actual_label "
        "FROM predictions ORDER BY predicted_at DESC LIMIT ?",
        (limit,),
    ).fetchall()

    result = []
    for transaction_id, features_json, score, decision, predicted_at, actual_label in rows:
        try:
            features = json.loads(features_json)
        except (TypeError, json.JSONDecodeError):
            features = {}
        result.append({
            "transaction_id": transaction_id,
            "user_id": features.get("user_id"),
            "amount": features.get("Amount"),
            "fraud_score": round(score, 4),
            "decision": decision,
            "predicted_at": predicted_at,
            "latency_ms": features.get("_latency_ms"),
            "actual_label": actual_label,
        })
    return result


@app.get("/live/summary")
def live_summary():
    """Aggregate KPIs (total counts, decision breakdown, avg score/latency) for the dashboard header."""
    total = feedback_store.conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    decision_counts = {"block": 0, "review": 0, "allow": 0}
    for decision, count in feedback_store.conn.execute(
        "SELECT decision, COUNT(*) FROM predictions GROUP BY decision"
    ).fetchall():
        decision_counts[decision] = count

    avg_score_row = feedback_store.conn.execute("SELECT AVG(fraud_score) FROM predictions").fetchone()
    avg_score = avg_score_row[0] if avg_score_row else None

    # latency sirf features JSON ke andar hai, isliye recent window Python mein parse karte hain
    recent_rows = feedback_store.conn.execute(
        "SELECT features FROM predictions ORDER BY predicted_at DESC LIMIT ?",
        (LIVE_WINDOW_SIZE,),
    ).fetchall()
    latencies = []
    for (features_json,) in recent_rows:
        try:
            features = json.loads(features_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if features.get("_latency_ms") is not None:
            latencies.append(features["_latency_ms"])
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    return {
        "total_transactions": total,
        "blocked": decision_counts["block"],
        "review": decision_counts["review"],
        "allowed": decision_counts["allow"],
        "avg_fraud_score": round(avg_score, 4) if avg_score is not None else None,
        "avg_latency_ms": round(avg_latency, 2) if avg_latency is not None else None,
    }


# ---------------------------------------------------------
# Dashboard serving
# ---------------------------------------------------------
# dashboard.html lives right next to api.py in src/, so it's served
# straight from here — reachable at http://<server>:8000/dashboard
# with no separate web server, and same-origin as the API (no CORS/HTTPS
# mismatch). Must come AFTER all @app.get/@app.post routes above.
from fastapi.responses import FileResponse

_DASHBOARD_PATH = os.path.join(BASE_DIR, "dashboard.html")


@app.get("/dashboard")
def serve_dashboard():
    return FileResponse(_DASHBOARD_PATH)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)