"""
ml/model.py — XGBoost burnout prediction engine
"""
import numpy as np

def predict_burnout_score(inputs: dict) -> float:
    """Simulate XGBoost inference. Replace with joblib.load('models/burnout_model.joblib')."""
    a  = inputs.get("attendance", 80)
    d  = inputs.get("delays", 2)
    g  = inputs.get("gpa", 3.2)
    s  = inputs.get("study", 18)
    e  = inputs.get("engagement", 60)
    em = inputs.get("emotional_score", 0.3)

    z = (0.5
         - 0.03 * (a - 75)
         + 0.12 * d
         - 0.25 * (3.0 - g)
         - 0.04 * (20 - s)
         - 0.08 * (e - 50)
         + 0.30 * (em - 0.5))
    score = 1 / (1 + np.exp(-z))
    return float(np.clip(score, 0.03, 0.97))


def get_feature_importance(inputs: dict) -> list:
    """Return SHAP-style feature importance list."""
    a  = inputs.get("attendance", 80)
    d  = inputs.get("delays", 2)
    g  = inputs.get("gpa", 3.2)
    s  = inputs.get("study", 18)
    e  = inputs.get("engagement", 60)
    em = inputs.get("emotional_score", 0.3)

    raw = {
        "Emotional Stress":   abs(0.30 * (em - 0.5)) + 0.05,
        "Assignment Delays":  abs(0.12 * d) + 0.04,
        "GPA Gap":            abs(0.25 * (3.0 - g)) + 0.03,
        "Low Attendance":     abs(0.03 * (a - 75)) + 0.02,
        "Study Hours":        abs(0.04 * (20 - s)) + 0.02,
        "Engagement":         abs(0.08 * (e - 50)) + 0.02,
    }
    total = sum(raw.values()) or 1
    return [
        {"feature": k, "importance": round(min(v / total, 0.98), 3)}
        for k, v in raw.items()
    ]
