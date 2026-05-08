import os
import sys
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from predict import predict_single

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "monitoring")
LOG_FILE = os.path.join(LOG_DIR, "prediction_log.jsonl")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Media Engagement Risk API",
    description="Scores media content for low-engagement risk using a trained sklearn pipeline.",
    version="1.0.0",
)


class ContentItem(BaseModel):
    media_type:           Optional[str]   = Field(None, example="video")
    entry_source:         Optional[str]   = Field(None, example="upload")
    status:               Optional[str]   = Field(None, example="Published")
    duration_msecs:       Optional[float] = Field(None, example=5_000_000)
    count_plays:          Optional[float] = Field(None, example=45)
    count_loads:          Optional[float] = Field(None, example=200)
    load_play_ratio:      Optional[float] = Field(None, example=0.225)
    unique_known_users:   Optional[float] = Field(None, example=8)
    unique_viewers:       Optional[float] = Field(None, example=30)
    sum_time_viewed:      Optional[float] = Field(None, example=50_000_000)
    avg_time_viewed:      Optional[float] = Field(None, example=1_666_667)
    avg_view_period_time: Optional[float] = Field(None, example=1_500_000)
    sum_view_period:      Optional[float] = Field(None, example=45_000_000)


class PredictionResponse(BaseModel):
    low_engagement_probability: float
    predicted_class:            int    # 0 = high engagement, 1 = low engagement
    risk_level:                 str    # Low / Medium / High
    model_version:              str = "1.0"
    scored_at:                  str


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/predict-risk", response_model=PredictionResponse)
def predict_risk(item: ContentItem):
    try:
        record = item.model_dump(exclude_none=False)
        result = predict_single(record)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not found. Run src/train.py first.")
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    scored_at = datetime.now(timezone.utc).isoformat()

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp":                   scored_at,
            "input_features":              record,
            "low_engagement_probability":  result["low_engagement_probability"],
            "predicted_class":             result["predicted_class"],
            "risk_level":                  result["risk_level"],
        }) + "\n")

    return PredictionResponse(
        low_engagement_probability=result["low_engagement_probability"],
        predicted_class=result["predicted_class"],
        risk_level=result["risk_level"],
        scored_at=scored_at,
    )
