# Real-Time Fraud Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Kafka](https://img.shields.io/badge/Apache%20Kafka-Streaming-black)
![FastAPI](https://img.shields.io/badge/FastAPI-API-teal)
![XGBoost](https://img.shields.io/badge/XGBoost-ML%20Model-orange)
![Feast](https://img.shields.io/badge/Feast-Feature%20Store-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![Redis](https://img.shields.io/badge/Redis-Online%20Store-dc382d)
![SQLite](https://img.shields.io/badge/SQLite-Feedback%20Store-lightgrey)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)
![MLOps](https://img.shields.io/badge/MLOps-Closed%20Loop-purple)

**Tags:** `fraud-detection` `real-time` `streaming` `kafka` `xgboost` `feast` `feature-store` `shap` `explainable-ai` `drift-detection` `fastapi` `redis` `sqlite` `docker` `mlops` `machine-learning`

A closed-loop, streaming fraud detection system for e-commerce and finance —
transactions are scored in milliseconds as they happen, not hours later in a
batch job.

> Banks and e-commerce platforms lose fraud detection value the moment a
> model only runs in batch: by the time a fraudulent transaction is flagged,
> the money is already gone. This project scores every transaction as it
> streams through the system and makes a block/review/allow decision before
> the customer's next click.

---

## Live demo

![Dashboard demo](assets/dashboard-demo.gif)

*(GIF: producer → Kafka → consumer → API → dashboard, updating live. See
[Setup and run order](#setup-and-run-order) to run it yourself.)*

---

## Architecture

```
kafka_producer.py                 kafka_consumer.py
  (simulates live      ──▶ Kafka ──▶  1. calls FastAPI /predict (XGBoost score)
   transaction stream)   topic:       2. pulls behavioral features from Feast
                        transactions  3. hybrid decision: ML score + velocity/
                                         amount-spike rules → block/review/allow
                                      4. logs every prediction to feedback.py (SQLite)
                                              │
                                              ▼
                                   ┌─────────────────────────┐
                                   │        FastAPI            │
                                   │  /predict   /explain       │
                                   │  /feedback  /health/metrics│
                                   │  /health/drift /retrain/export
                                   │  /live/transactions /live/summary
                                   └─────────────────────────┘
                                              │
                                              ▼
                                  dashboard.html (live, polls every 2s)
                                  ── KPI cards · live transaction table ──
                                  ── decision breakdown · model health  ──

Analyst confirms fraud/legit ──▶ POST /feedback ──▶ /health/metrics (real precision/recall)
                                                  └─▶ /retrain/export (CSV for next training run)
```

## Key features

| Feature | Where | Why it matters |
|---|---|---|
| **Sub-second scoring** | `api.py` `/predict` | XGBoost model returns a fraud probability + decision in milliseconds, not a batch job hours later. |
| **Hybrid decision engine** | `kafka_consumer.py` | Combines the ML score with behavioral rules (transaction velocity, amount-spike detection) — catches patterns a pure ML score can miss. |
| **Real-time feature store** | `feature_store.py` + `feature_repo/` | Behavioral features (5-min transaction count, amount ratio, velocity/spike flags) are served through **Feast**, so training and serving use the same feature logic — no train/serve skew. |
| **Explainability** | `/explain` (SHAP) | Every score comes with *why* — the features driving it. Required in any real fraud/compliance workflow, not just a nice-to-have. |
| **Drift monitoring** | `/health/drift` (PSI) | Detects when live traffic statistically diverges from training data — the most common, silent way production fraud models degrade. |
| **Closed feedback loop** | `/feedback`, `/health/metrics`, `/retrain/export` | Captures ground-truth outcomes after the fact, computes *real* precision/recall (not training-set numbers), and exports a retrain-ready dataset. Most portfolio projects stop at "model predicts, done" — this one closes the loop back to retraining. |
| **Live operational dashboard** | `dashboard.html` | Self-contained HTML/CSS/JS console (no build step) polling the API every 2s: live transaction feed, decision breakdown, and model health, all in one view. |

## Tech stack

- **Streaming:** Apache Kafka (Confluent images), Zookeeper
- **Modeling:** XGBoost (supervised) — trained on the Kaggle credit-card fraud dataset
- **Feature store:** Feast (online: Redis, offline: Parquet)
- **Serving:** FastAPI + Pydantic
- **Explainability:** SHAP
- **Drift detection:** PSI (Population Stability Index) against a saved training baseline
- **Feedback / retraining data:** SQLite (`feedback.py`)
- **Dashboard:** Vanilla HTML/CSS/JS (no framework, no build step)

## Project layout

```
real-time-fraud-detection-pipeline/
├── data/
│   ├── creditcard.csv                 # Kaggle dataset (not committed — see Setup)
│   ├── baseline_distributions.json    # created by train_model.py
│   ├── feedback.db                    # created automatically on first prediction
│   └── retrain_dataset.csv            # created by /retrain/export
├── models/                            # created by train_model.py
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
├── feature_repo/                      # Feast repo
│   ├── feature_store.yaml
│   ├── entities.py
│   └── features.py
├── src/
│   ├── train_model.py
│   ├── feature_store.py               # Feast-backed behavioral features
│   ├── explainability.py              # SHAP
│   ├── drift_monitor.py               # PSI drift detection
│   ├── feedback.py                    # closed-loop feedback store (SQLite)
│   ├── api.py                         # /predict /explain /feedback /health/* /retrain/export /live/*
│   ├── kafka_producer.py              # simulated live transaction stream
│   ├── kafka_consumer.py              # scores, applies hybrid rules, logs feedback
│   └── dashboard.html                 # live operational dashboard
├── docker-compose.yml                 # Kafka + Zookeeper + Redis
└── requirements.txt
```

## Setup and run order

```bash
pip install -r requirements.txt

# 1. Start Kafka + Zookeeper + Redis
docker compose up -d

# 2. Initialize Feast (once)
cd feature_repo && feast apply && cd ..

# 3. Train the model — also saves the drift-detection baseline
cd src
python train_model.py

# 4. Start the API
uvicorn api:app --reload --port 8000

# 5. In two separate terminals, start the consumer and producer
python kafka_consumer.py
python kafka_producer.py
```

Then open `src/dashboard.html` in a browser — it polls the API automatically
and starts showing live transactions within a couple of seconds.

> The Kaggle **Credit Card Fraud Detection** dataset (`creditcard.csv`,
> ~150MB, [available here](https://www.kaggle.com/mlg-ulb/creditcardfraud))
> is optional. If it's not present in `data/`, the producer generates
> synthetic transactions instead — the pipeline runs either way, though
> fraud scores are more realistic with the real dataset.

## 5-minute interview demo

1. Let the producer/consumer run for a minute so transactions flow through
   and the dashboard fills up.
2. `POST /explain` on a transaction — walk through the SHAP breakdown of
   why it scored the way it did.
3. Temporarily scale up the amount distribution in `kafka_producer.py`
   (e.g. 5× normal transaction amounts) and hit `GET /health/drift` — watch
   it move to `significant_shift`.
4. `POST /feedback` with a few `{transaction_id, actual_label}` pairs
   (grab `transaction_id` values from the consumer logs or the dashboard),
   then `GET /health/metrics` — show real precision/recall computed from
   confirmed outcomes, not training-set numbers.
5. `POST /retrain/export` — show the CSV that drops out, ready to feed
   straight back into `train_model.py`.

This walkthrough demonstrates the full MLOps lifecycle — score, explain,
monitor, collect feedback, retrain — not just a trained model sitting
behind a single endpoint.

## Design notes

- **Why a hybrid decision engine, not just the ML score?** Pure ML scores
  can miss velocity-based fraud (many small transactions in a short window)
  that a single-transaction model was never trained to see. Combining the
  model's score with lightweight behavioral rules catches both.
- **Why Feast instead of a plain Redis hash?** It keeps the exact feature
  computation used at serving time versioned and reusable for offline
  training — avoiding the train/serve skew that quietly breaks a lot of
  real fraud models.
- **Why SQLite for feedback, not Postgres?** This is a single-node demo
  pipeline; SQLite keeps setup to zero extra services. The feedback store
  is isolated behind `feedback.py`, so swapping in Postgres later is a
  contained change.
