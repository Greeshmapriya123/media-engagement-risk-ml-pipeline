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

from sklearn.metrics import roc_curve, auc

ROOT        = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR  = os.path.join(ROOT, "models")
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")


def get_feature_names(pipeline) -> list:
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        return []


def plot_feature_importance(pipeline, model_name: str, top_n: int = 20):
    clf   = pipeline.named_steps["classifier"]
    names = get_feature_names(pipeline)
    if not names:
        return

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        idx  = np.argsort(importances)[::-1][:top_n]
        vals = importances[idx]
        feat = [names[i] if i < len(names) else f"f{i}" for i in idx]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(feat)))
        ax.barh(feat[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
        ax.set_title(f"Feature Importances — {model_name}")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        _save(fig, "08_feature_importance.png")

    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        idx  = np.argsort(np.abs(coefs))[::-1][:top_n]
        vals = coefs[idx]
        feat = [names[i] if i < len(names) else f"f{i}" for i in idx]

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
        colors = ["#F44336" if v > 0 else "#4CAF50" for v in vals[::-1]]
        ax.barh(feat[::-1], vals[::-1], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Logistic Regression Coefficients — {model_name}")
        ax.set_xlabel("Coefficient  (positive → increases low-engagement probability)")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        _save(fig, "08_lr_coefficients.png")


def plot_roc_curve(pipeline, X_test, y_test, model_name: str):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2196F3")
    ax.set_title(f"ROC Curve — {model_name} (test set)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "09_roc_curve.png")


def print_top_drivers(pipeline, model_name: str):
    """
    Print the top predictive features in plain language.
    These are correlations observed in training data, not causal claims.
    """
    clf   = pipeline.named_steps["classifier"]
    names = get_feature_names(pipeline)
    if not names:
        return

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        return

    top5 = sorted(
        enumerate(importances), key=lambda x: x[1], reverse=True
    )[:5]

    label_map = {
        "num__load_play_ratio":      "Low load-to-play ratio (viewers load but don't press play)",
        "num__viewer_play_ratio":    "Low viewer-play ratio (content not converting page loads)",
        "num__play_intensity":       "Low re-watch rate (viewers not returning)",
        "num__known_user_share":     "Low share of authenticated users",
        "num__duration_mins":        "Content duration",
        "num__count_plays":          "Total play count",
        "num__unique_viewers":       "Unique viewer count",
        "num__log_count_plays":      "Play volume (log scale)",
        "num__log_unique_viewers":   "Audience size (log scale)",
        "num__avg_time_viewed":      "Average time viewed per session",
    }

    print("\nTop predictive features:")
    for rank, (i, score) in enumerate(top5, 1):
        raw_name = names[i] if i < len(names) else f"feature_{i}"
        human    = label_map.get(raw_name, raw_name.replace("num__", "").replace("cat__", "").replace("_", " "))
        print(f"  {rank}. {human}  (importance={score:.4f})")

    print(
        "\nNote: these are correlations in the training data. "
        "They do not imply causation."
    )


def run_evaluation(X_test=None, y_test=None):
    model_path    = os.path.join(MODELS_DIR, "best_pipeline.pkl")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.pkl")

    if not os.path.exists(model_path):
        print("No saved model found. Run train.py first.")
        return

    pipeline   = joblib.load(model_path)
    meta       = joblib.load(metadata_path)
    model_name = meta["model_name"]

    print(f"Loaded model: {model_name}")

    if X_test is not None and y_test is not None:
        plot_roc_curve(pipeline, X_test, y_test, model_name)

    plot_feature_importance(pipeline, model_name)
    print_top_drivers(pipeline, model_name)


if __name__ == "__main__":
    from data_load import load_raw, drop_leakage_columns, build_target
    from features import engineer_features, get_feature_lists
    from sklearn.model_selection import train_test_split

    raw = load_raw()
    df  = drop_leakage_columns(raw)
    df  = build_target(df)
    df  = engineer_features(df)
    num_cols, cat_cols = get_feature_lists(df)
    X = df[num_cols + cat_cols]
    y = df["is_low_engagement"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    run_evaluation(X_test, y_test)
