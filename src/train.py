import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

import mlflow
import mlflow.sklearn

from data_load import load_raw, drop_leakage_columns, build_target
from features import engineer_features, get_feature_lists, build_preprocessor

ROOT        = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR  = os.path.join(ROOT, "models")
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

mlflow.set_tracking_uri(os.path.join(ROOT, "mlruns"))

EXPERIMENT  = "media-engagement-risk"
TARGET      = "is_low_engagement"
RANDOM_SEED = 42


def compute_metrics(model, X, y, prefix: str) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_roc_auc":   round(roc_auc_score(y, y_prob), 4),
        f"{prefix}_f1":        round(f1_score(y, y_pred), 4),
        f"{prefix}_recall":    round(recall_score(y, y_pred), 4),
        f"{prefix}_precision": round(precision_score(y, y_pred, zero_division=0), 4),
    }


def save_confusion_matrix(model, X, y, name: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y, model.predict(X))
    ConfusionMatrixDisplay(cm, display_labels=["High Eng.", "Low Eng."]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {name}")
    path = os.path.join(FIGURES_DIR, f"cm_{name.lower().replace(' ', '_')}.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def cross_val_metrics(model, X, y, cv: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    results = cross_validate(
        model, X, y, cv=skf,
        scoring=["roc_auc", "f1", "recall", "precision"],
        n_jobs=-1,
    )
    out = {}
    for metric in ["roc_auc", "f1", "recall", "precision"]:
        vals = results[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = round(vals.mean(), 4)
        out[f"cv_{metric}_std"]  = round(vals.std(), 4)
    return out


def train():
    print("Loading data...")
    raw = load_raw()
    df  = drop_leakage_columns(raw)
    df  = build_target(df)
    df  = engineer_features(df)
    num_cols, cat_cols = get_feature_lists(df)

    X = df[num_cols + cat_cols].copy()
    y = df[TARGET].copy()
    print(f"Features: {X.shape[1]}   Samples: {len(X)}")
    print(f"Class split:\n{y.value_counts()}\n")

    # 60 / 20 / 20 stratified split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}\n")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_weight = neg / max(pos, 1)

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        ),
    }
    if LGBM_AVAILABLE:
        candidates["LightGBM"] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=scale_weight, random_state=RANDOM_SEED,
            verbosity=-1, n_jobs=-1,
        )
    else:
        candidates["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            random_state=RANDOM_SEED,
        )

    mlflow.set_experiment(EXPERIMENT)
    results = {}

    for name, clf in candidates.items():
        print(f"Training: {name}")
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            val_metrics = compute_metrics(pipeline, X_val, y_val, prefix="val")
            cv_metrics  = cross_val_metrics(pipeline, X_tv, y_tv)

            mlflow.log_params(clf.get_params())
            mlflow.log_metrics({**val_metrics, **cv_metrics})
            mlflow.log_artifact(save_confusion_matrix(pipeline, X_val, y_val, name))
            mlflow.sklearn.log_model(pipeline, artifact_path="pipeline")

            results[name] = {
                "pipeline": pipeline,
                "val_metrics": val_metrics,
                "cv_metrics": cv_metrics,
                "run_id": mlflow.active_run().info.run_id,
            }

        print(
            f"  val  AUC={val_metrics['val_roc_auc']}  "
            f"F1={val_metrics['val_f1']}  "
            f"Recall={val_metrics['val_recall']}  "
            f"Precision={val_metrics['val_precision']}"
        )
        print(
            f"  cv   AUC={cv_metrics['cv_roc_auc_mean']}±{cv_metrics['cv_roc_auc_std']}  "
            f"Recall={cv_metrics['cv_recall_mean']}±{cv_metrics['cv_recall_std']}\n"
        )

    # Select best by recall, then AUC
    best_name = max(
        results,
        key=lambda n: (
            results[n]["val_metrics"]["val_recall"],
            results[n]["val_metrics"]["val_roc_auc"],
        ),
    )
    best = results[best_name]
    print(f"Best model: {best_name}\n")

    # Final test evaluation — run once only
    pipeline = best["pipeline"]
    y_pred   = pipeline.predict(X_test)
    y_prob   = pipeline.predict_proba(X_test)[:, 1]

    test_metrics = {
        "test_roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        "test_f1":        round(f1_score(y_test, y_pred), 4),
        "test_recall":    round(recall_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
    }

    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print()
    print(classification_report(y_test, y_pred, target_names=["High Eng.", "Low Eng."]))

    save_confusion_matrix(pipeline, X_test, y_test, f"{best_name}_test")

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics(test_metrics)
        mlflow.set_tag("best_model", "true")

    joblib.dump(pipeline, os.path.join(MODELS_DIR, "best_pipeline.pkl"))
    joblib.dump(
        {
            "model_name": best_name,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "test_metrics": test_metrics,
            "target": TARGET,
        },
        os.path.join(MODELS_DIR, "model_metadata.pkl"),
    )
    print(f"Pipeline saved to models/best_pipeline.pkl")


if __name__ == "__main__":
    train()
