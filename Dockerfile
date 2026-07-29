
# Deploys only api.py + explainability.py + drift_monitor.py + feedback.py
# No Kafka/Redis/Feast needed — those are only used by kafka_consumer.py,
# which stays local for the full-pipeline demo.

FROM python:3.11-slim

WORKDIR /app

COPY requirements-api-deploy.txt .
RUN pip install --no-cache-dir -r requirements-api-deploy.txt

# Only what api.py actually needs
COPY src/api.py src/
COPY src/explainability.py src/
COPY src/drift_monitor.py src/
COPY src/feedback.py src/
COPY models/ models/
COPY data/baseline_distributions.json data/baseline_distributions.json

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
