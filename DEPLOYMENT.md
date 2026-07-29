# Deployment Guide — Live Public Demo (Free)

This deploys two pieces:
1. **The API** (model + explainability + drift + feedback) → Render.com, free tier
2. **The dashboard** (live simulated traffic + charts) → Streamlit Community Cloud, free tier

Neither needs Kafka or Redis running in the cloud — those stay local, for the
full-pipeline demo you already have working. The deployed API is self-contained.

---

## Part 1 — Push your project to GitHub

If you haven't already:
```bash
cd fraud-project-merged
git init
git add .
git commit -m "Fraud detection MLOps pipeline"
```
Create a new repo on https://github.com/new, then:
```bash
git remote add origin https://github.com/YOUR-USERNAME/fraud-detection-mlops.git
git branch -M main
git push -u origin main
```

**Important:** `data/creditcard.csv` is large (~150MB) — GitHub will reject it over 100MB.
Add it to `.gitignore` before committing (see below), and don't push it. The deployed
API doesn't need it — only `models/` and `data/baseline_distributions.json` do.

Create a `.gitignore`:
```
venv/
data/creditcard.csv
data/feedback.db
data/retrain_dataset.csv
data/user_behavior_features.parquet
__pycache__/
*.pyc
```

---

## Part 2 — Deploy the API to Render

1. Go to https://render.com, sign up (free), click **New +** → **Web Service**
2. Connect your GitHub repo
3. Render should auto-detect `render.yaml` and `Dockerfile` — if not, set manually:
   - **Environment:** Docker
   - **Dockerfile path:** `./Dockerfile`
   - **Plan:** Free
4. Click **Create Web Service**. First build takes ~5 minutes.
5. Once live, you'll get a URL like `https://fraud-detection-api-xxxx.onrender.com`
6. Test it: open `https://YOUR-URL.onrender.com/docs` — Swagger UI should load.

**Free tier note:** the service sleeps after 15 minutes of no traffic, and the first
request after that takes 30-60 seconds to wake up. This is normal and fine for a
portfolio demo — the dashboard has a note about this built in.

---

## Part 3 — Deploy the dashboard to Streamlit Cloud

1. In `dashboard/app.py`, replace this line with your actual Render URL:
   ```python
   DEFAULT_API_URL = "https://YOUR-RENDER-APP.onrender.com"
   ```
   Commit and push this change.

2. Go to https://share.streamlit.io, sign in with GitHub
3. Click **New app**, select your repo, set:
   - **Main file path:** `dashboard/app.py`
4. Click **Deploy**. Takes ~2 minutes.

You'll get a public URL like `https://your-app.streamlit.app` — this is what you
put on your resume/portfolio/LinkedIn. Anyone can open it and watch live fraud
predictions happen in real time.

---

## What to put on your resume/portfolio

> Live demo: https://your-app.streamlit.app
> Source: https://github.com/YOUR-USERNAME/fraud-detection-mlops

And in the project description, mention: real-time XGBoost fraud scoring (98%
ROC-AUC on the Kaggle credit card fraud dataset), SHAP explainability, PSI-based
drift monitoring, and a closed feedback loop for continuous retraining — deployed
as a live public API with a Streamlit monitoring dashboard.

---

## Optional: also show the full Kafka pipeline

Since Kafka/Redis/Feast aren't in the public deployment, record a 1-2 minute
screen recording of your local full pipeline running (producer → consumer →
Feast → decisions in the terminal) and link it in your README/portfolio. This
covers the parts of the architecture the live demo can't show for free.
