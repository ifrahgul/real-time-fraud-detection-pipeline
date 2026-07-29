"""
Explainability — turns a raw fraud score into a reason, using SHAP
TreeExplainer (exact and fast for XGBoost).
"""

import shap
import pandas as pd


class Explainer:
    def __init__(self, model, top_k=5):
        self.model = model
        self.top_k = top_k
        self.explainer = shap.TreeExplainer(model)

    def explain(self, feature_row: dict):
        """
        feature_row: dict of {feature_name: value}, same shape/order used for /predict
        (already scaled, matching what the model actually sees).
        """
        df = pd.DataFrame([feature_row])
        shap_values = self.explainer.shap_values(df)

        contributions = list(zip(df.columns, shap_values[0]))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_features = [
            {
                "feature": name,
                "value": float(df[name].iloc[0]),
                "impact": round(float(impact), 4),
                "direction": "pushes toward fraud" if impact > 0 else "pushes toward legitimate",
            }
            for name, impact in contributions[: self.top_k]
        ]

        base_value = float(self.explainer.expected_value)
        prediction = float(self.model.predict_proba(df)[0][1])

        return {
            "fraud_probability": round(prediction, 4),
            "base_rate": round(base_value, 4),
            "top_contributing_features": top_features,
        }
