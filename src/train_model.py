"""
Step 1: Fraud Detection Model Training
========================================
Yeh script Kaggle 'Credit Card Fraud Detection' dataset par XGBoost model
train karti hai. Agar dataset nahi mila to yeh khud synthetic fraud data
generate kar leti hai taake aap turant test kar sakein.

Extended: training ke baad ek drift baseline bhi save karti hai (PSI ke
liye) — taake production mein feature drift detect kiya ja sake.

Real dataset download karne ke liye (recommended):
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    -> creditcard.csv ko is folder mein rakh dein.
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from drift_monitor import DriftMonitor

import os as _os

# Yeh script src/ folder ke andar se chalti hai, isliye paths ko
# ek level upar (project root) ke data/ aur models/ folders se link kiya hai.
BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
PROJECT_ROOT = _os.path.dirname(BASE_DIR)

DATA_DIR = _os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = _os.path.join(PROJECT_ROOT, "models")
_os.makedirs(MODELS_DIR, exist_ok=True)

DATA_PATH = _os.path.join(DATA_DIR, "creditcard.csv")
MODEL_PATH = _os.path.join(MODELS_DIR, "fraud_model.pkl")
SCALER_PATH = _os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = _os.path.join(MODELS_DIR, "metrics.json")
FEATURES_PATH = _os.path.join(MODELS_DIR, "feature_names.json")

RANDOM_STATE = 42


# ---------------------------------------------------------
# 1. DATA LOADING (real dataset ya synthetic fallback)
# ---------------------------------------------------------
def load_data():
    if os.path.exists(DATA_PATH):
        print(f"[INFO] Real dataset mil gaya: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("[WARNING] creditcard.csv nahi mila. Synthetic dataset generate kar rahe hain "
              "taake aap pipeline turant test kar sakein. Real accuracy ke liye Kaggle se "
              "asli dataset download kar ke is folder mein rakhein.")
        df = generate_synthetic_data(n_samples=50000, fraud_ratio=0.0017)
    return df


def generate_synthetic_data(n_samples=50000, fraud_ratio=0.0017):
    """
    Kaggle dataset jaisa structure: Time, Amount, V1-V28 (PCA features), Class
    Fraud transactions ko thora shift kiya gaya hai taake model kuch seekh sake.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    n_fraud = max(int(n_samples * fraud_ratio), 20)
    n_normal = n_samples - n_fraud

    n_features = 28  # V1...V28

    normal = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_features))
    normal_amount = np.abs(rng.normal(loc=60, scale=80, size=n_normal))
    normal_time = rng.uniform(0, 172800, size=n_normal)

    fraud = rng.normal(loc=1.5, scale=2.5, size=(n_fraud, n_features))
    fraud_amount = np.abs(rng.normal(loc=350, scale=250, size=n_fraud))
    fraud_time = rng.uniform(0, 172800, size=n_fraud)

    X = np.vstack([normal, fraud])
    amount = np.concatenate([normal_amount, fraud_amount])
    time = np.concatenate([normal_time, fraud_time])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    cols = [f"V{i}" for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = labels.astype(int)

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


# ---------------------------------------------------------
# 2. PREPROCESSING
# ---------------------------------------------------------
def preprocess(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    return X, y, scaler


# ---------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------
def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


# ---------------------------------------------------------
# 4. MODEL TRAINING (class imbalance handle karte hue)
# ---------------------------------------------------------
def train_model(X_train, y_train):
    fraud_count = y_train.sum()
    normal_count = len(y_train) - fraud_count
    scale_pos_weight = normal_count / fraud_count

    print(f"[INFO] Fraud cases: {fraud_count}, Normal cases: {normal_count}, "
          f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------
# 5. EVALUATION (accuracy nahi, precision/recall/F1 dekhte hain)
# ---------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print("Confusion Matrix:")
    print(cm)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Average Precision (PR-AUC): {avg_precision:.4f}")
    print("=" * 50)

    metrics = {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "precision_fraud": report["Fraud"]["precision"],
        "recall_fraud": report["Fraud"]["recall"],
        "f1_fraud": report["Fraud"]["f1-score"],
        "confusion_matrix": cm.tolist(),
    }
    return metrics


# ---------------------------------------------------------
# 6. SAVE MODEL + SCALER
# ---------------------------------------------------------
def save_artifacts(model, scaler, metrics, feature_names):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(FEATURES_PATH, "w") as f:
        json.dump(list(feature_names), f)

    print(f"\n[SAVED] Model -> {MODEL_PATH}")
    print(f"[SAVED] Scaler -> {SCALER_PATH}")
    print(f"[SAVED] Metrics -> {METRICS_PATH}")
    print(f"[SAVED] Feature names -> {FEATURES_PATH}")


# ---------------------------------------------------------
# 7. SAVE DRIFT BASELINE (new — for /health/drift in api.py)
# ---------------------------------------------------------
def save_drift_baseline(raw_df):
    """
    Saves the training-time distribution of the RAW (unscaled) features —
    this must match what kafka_consumer.py actually logs in production
    (raw Amount/Time straight from the transaction, not scaler-transformed
    values), or every comparison will show false "drift" just from the
    scale mismatch.
    """
    drift_columns = ["Amount", "Time", "V1", "V2", "V3", "V4", "V14", "V17"]
    drift_columns = [c for c in drift_columns if c in raw_df.columns]
    baseline_path = os.path.join(DATA_DIR, "baseline_distributions.json")
    DriftMonitor.save_baseline(raw_df, columns=drift_columns, path=baseline_path)
    print(f"[SAVED] Drift baseline -> {baseline_path} (columns: {drift_columns})")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    print("[STEP 1] Loading data...")
    df = load_data()
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Fraud cases: {df['Class'].sum()} / {len(df)} "
          f"({df['Class'].mean() * 100:.3f}%)")

    print("\n[STEP 2] Preprocessing...")
    X, y, scaler = preprocess(df)

    print("\n[STEP 3] Splitting train/test...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    print("\n[STEP 4] Training XGBoost model...")
    model = train_model(X_train, y_train)

    print("\n[STEP 5] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print("\n[STEP 6] Saving model artifacts...")
    save_artifacts(model, scaler, metrics, X.columns)

    print("\n[STEP 7] Saving drift baseline...")
    save_drift_baseline(df)

    print("\n✅ Step 1 complete! Ab aapke paas ek trained fraud detection model hai.")
    print("   Next step: is model ko FastAPI ke zariye ek prediction endpoint banayenge.")


if __name__ == "__main__":
    main()