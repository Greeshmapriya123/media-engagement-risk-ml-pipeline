import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import engineer_features

ROOT          = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH    = os.path.join(ROOT, "models", "best_pipeline.pkl")
METADATA_PATH = os.path.join(ROOT, "models", "model_metadata.pkl")


def load_model():
    pipeline = joblib.load(MODEL_PATH)
    meta     = joblib.load(METADATA_PATH)
    return pipeline, meta


def engagement_risk_level(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    return "High"


def predict_single(record: dict) -> dict:
    pipeline, meta = load_model()
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    df = pd.DataFrame([record])
    df = engineer_features(df)

    for col in num_cols + cat_cols:
        if col not in df.columns:
            df[col] = np.nan

    X    = df[num_cols + cat_cols]
    prob = float(pipeline.predict_proba(X)[0, 1])
    pred = int(pipeline.predict(X)[0])

    return {
        "low_engagement_probability": round(prob, 4),
        "predicted_class": pred,
        "risk_level": engagement_risk_level(prob),
    }


def predict_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    pipeline, meta = load_model()
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]

    df = engineer_features(df_input.copy())
    for col in num_cols + cat_cols:
        if col not in df.columns:
            df[col] = np.nan

    X     = df[num_cols + cat_cols]
    probs = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)

    out = df_input.copy()
    out["low_engagement_probability"] = probs.round(4)
    out["predicted_class"]            = preds
    out["risk_level"]                 = [engagement_risk_level(p) for p in probs]
    return out


if __name__ == "__main__":
    sample = {
        "media_type": "video",
        "entry_source": "upload",
        "status": "Published",
        "duration_msecs": 5_000_000,
        "count_plays": 45,
        "count_loads": 200,
        "load_play_ratio": 0.225,
        "unique_known_users": 8,
        "unique_viewers": 30,
        "sum_time_viewed": 50_000_000,
        "avg_time_viewed": 1_666_667,
        "avg_view_period_time": 1_500_000,
        "sum_view_period": 45_000_000,
    }
    print(predict_single(sample))
