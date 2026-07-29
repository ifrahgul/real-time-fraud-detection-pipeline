# Fraud Detection MLOps Pipeline — Closed-Loop Edition

```
Producer → Kafka → Consumer ──┬─→ Feast (feature store: online + offline)
                               ├─→ FastAPI /predict → decision (block/review/allow)
                               └─→ feedback.py: logs every prediction

Analyst confirms fraud/legit ──→ POST /feedback ──→ /health/metrics (real precision/recall)
                                                   └→ /retrain/export (CSV for next training run)

FastAPI /explain      → SHAP: why this transaction got this score
FastAPI /health/drift → PSI check: has live traffic drifted from training data?
```

## Project layout

```
project_root/
├── data/                       # creditcard.csv goes here (optional, else synthetic data is used)
│   ├── baseline_distributions.json   # created by train_model.py, step 7
│   ├── feedback.db                    # created automatically on first prediction
│   └── retrain_dataset.csv            # created by /retrain/export
├── models/                     # created by train_model.py
├── feature_repo/               # Feast repo
│   ├── feature_store.yaml
│   ├── entities.py
│   └── features.py
└── src/
    ├── train_model.py
    ├── feature_store.py        # now Feast-backed (was raw Redis)
    ├── explainability.py       # new: SHAP
    ├── drift_monitor.py        # new: PSI drift detection
    ├── feedback.py             # new: closed-loop feedback store
    ├── api.py                  # extended: /explain, /feedback, /health/metrics, /health/drift, /retrain/export
    ├── kafka_producer.py       # unchanged
    └── kafka_consumer.py       # extended: logs every prediction to feedback store
```

## Setup and run order

```bash
pip install -r requirements.txt

# 1. Start Redis + Kafka (your existing docker-compose.yml)
docker compose up -d

# 2. Initialize Feast (once)
cd feature_repo && feast apply && cd ..

# 3. Train the model — this now also saves a drift baseline
cd src
python train_model.py

# 4. Start the API
uvicorn api:app --reload --port 8000

# 5. In separate terminals, start the consumer and producer
python kafka_consumer.py
python kafka_producer.py
```

## What's new, and why it matters for a portfolio

- **Feast feature store** (`feature_store.py`, `feature_repo/`) — behavioral features are now versioned and served with training/serving consistency, not just held in a raw Redis hash.
- **SHAP explainability** (`/explain`) — every score comes with a reason, which real fraud/compliance teams require.
- **Drift monitoring** (`/health/drift`) — PSI-based check for whether live traffic still resembles training data. This is the actual, most common reason production fraud models silently degrade.
- **Closed feedback loop** (`/feedback`, `/health/metrics`, `/retrain/export`) — captures ground truth after the fact, computes real precision/recall (not training-set numbers), and exports a retrain-ready dataset. Most portfolio projects stop at "model predicts, done" — this one closes the loop.

## 5-minute interview demo

1. Let the producer/consumer run for a minute so some transactions flow through.
2. `POST /explain` on a transaction — show the SHAP breakdown of why it scored the way it did.
3. Temporarily bump `fraud_probability` in `kafka_producer.py`'s amount distribution (e.g. 5x normal amounts) and hit `GET /health/drift` — watch it move to `significant_shift`.
4. `POST /feedback` with a few `{transaction_id, actual_label}` pairs (grab `transaction_id` from consumer logs), then `GET /health/metrics` — show real precision/recall.
5. `POST /retrain/export` — show the CSV that drops out, ready for the next `train_model.py` run.

That walkthrough demonstrates the full MLOps lifecycle — not just a trained model behind an endpoint.
