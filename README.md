# Media Engagement Risk Pipeline

A classification pipeline that predicts whether a piece of media content is at risk of low engagement (below 40% average completion rate). The project covers the full ML workflow: data preparation, feature engineering, model training with MLflow tracking, a FastAPI inference service, Docker packaging, and a data drift monitoring script.

---

## Dataset

The dataset is a synthetic demo CSV (`data/media_analytics.csv`) with 100 rows and 20 columns representing media content items — videos, audio files, and webcasts. It includes view counts, load counts, completion rates, and source/type metadata.

Because the dataset is small and synthetic, the model metrics on the test set are not reliable performance estimates. The purpose of this project is to demonstrate a complete, well-structured ML pipeline, not to claim strong predictive performance.

---

## Problem framing

The target variable `is_low_engagement` is derived from `avg_completion_rate`:

```
is_low_engagement = 1  if avg_completion_rate < 0.40
is_low_engagement = 0  otherwise
```

The threshold (0.40) is a configurable business parameter set before any train/test split. It does not cause leakage.

---

## Data leakage

Three columns are dropped before any modelling:

| Column | Where dropped | Reason |
|---|---|---|
| `engagement_ranking` | `data_load.py` | A composite rank computed from the final engagement outcome — it directly encodes what we're predicting. |
| `avg_completion_rate` | `features.py` | Used to build the target label. Kept in the DataFrame until the label is created, then excluded from the feature matrix. |
| `avg_view_drop_off` | `features.py` | The inverse of `avg_completion_rate`. Keeping it would be equivalent to leaking the target. |

---

## Feature engineering

Derived features added on top of the raw columns:

| Feature | Formula |
|---|---|
| `duration_mins` | `duration_msecs / 60000` |
| `viewer_play_ratio` | `unique_viewers / (count_loads + 1)` |
| `known_user_share` | `unique_known_users / (unique_viewers + 1)` |
| `play_intensity` | `count_plays / (unique_viewers + 1)` |
| `log_count_plays` | `log1p(count_plays)` |
| `log_sum_time_viewed` | `log1p(sum_time_viewed)` |

---

## Models

Three models are trained and compared. Model selection is based on recall first (missing an at-risk item is more costly than a false alarm), then ROC-AUC.

- Logistic Regression (baseline)
- Random Forest
- LightGBM (falls back to GradientBoostingClassifier if LightGBM is not installed)

Each model is wrapped in a scikit-learn `Pipeline` with a `ColumnTransformer` that handles imputation, scaling, and one-hot encoding.

---

## Test results (100-row synthetic dataset)

These numbers are from a 20-item test set and should be treated as illustrative only.

| Metric | Score |
|---|---|
| ROC-AUC | 0.58 |
| Recall (low engagement) | 0.38 |
| Precision (low engagement) | 1.00 |
| F1 (low engagement) | 0.55 |

A test set of 20 items produces high-variance estimates. Results would stabilise with more data.

---

## Project structure

```
media-engagement-ml-pipeline/
├── data/
│   └── media_analytics.csv
├── src/
│   ├── data_load.py       # loading, leakage removal, target creation
│   ├── features.py        # feature engineering and sklearn preprocessor
│   ├── eda.py             # EDA plots saved to reports/figures/
│   ├── train.py           # training loop with MLflow tracking
│   ├── evaluate.py        # ROC curve, feature importance, top drivers
│   └── predict.py         # single-item and batch inference
├── app/
│   └── main.py            # FastAPI service
├── models/
│   ├── best_pipeline.pkl
│   └── model_metadata.pkl
├── reports/
│   └── figures/
├── monitoring/
│   └── drift_report.py    # Evidently AI drift report
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## How to run

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Train**

```bash
python src/train.py
```

Trains all three models, logs runs to MLflow, and saves the best pipeline to `models/`.

**EDA plots**

```bash
python src/eda.py
# plots saved to reports/figures/
```

**MLflow UI**

```bash
mlflow ui --backend-store-uri mlruns/
# open http://localhost:5000
```

**API**

```bash
uvicorn app.main:app --reload --port 8000
# docs at http://localhost:8000/docs
```

**Example request**

```bash
curl -X POST http://localhost:8000/predict-risk \
  -H "Content-Type: application/json" \
  -d '{
    "media_type": "video",
    "entry_source": "upload",
    "status": "Published",
    "duration_msecs": 5000000,
    "count_plays": 45,
    "count_loads": 200,
    "load_play_ratio": 0.225,
    "unique_known_users": 8,
    "unique_viewers": 30
  }'
```

```json
{
  "low_engagement_probability": 0.72,
  "predicted_class": 1,
  "risk_level": "High",
  "model_version": "1.0",
  "scored_at": "2025-05-08T14:00:00Z"
}
```

---

## Docker

```bash
# Build (run train.py first so models/ is populated)
docker build -t media-risk-api .

# Run
docker run -p 8000:8000 media-risk-api
```

---

## Monitoring

Every `/predict-risk` call is appended to `monitoring/prediction_log.jsonl` with the input features, predicted class, probability, and timestamp.

To generate a drift report comparing training data against logged production requests:

```bash
python monitoring/drift_report.py
# opens reports/drift_report.html
```

Key things to watch:
- Per-feature Jensen-Shannon distance above 0.10
- More than 20% of features drifting simultaneously
- The predicted at-risk rate shifting more than 5 percentage points from the training baseline

---

## Possible next steps

- Replace the synthetic data with a real content analytics export
- Add time-based train/test splits to simulate deployment conditions
- Add content metadata (topic, target audience) as features
- Tune the 0.40 completion threshold using a precision-recall curve
- Add SHAP values for per-prediction explanations
