import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_class_distribution(df: pd.DataFrame, target: str = "is_low_engagement"):
    counts = df[target].value_counts().sort_index()
    labels = ["High Engagement (0)", "Low Engagement (1)"]
    colors = ["#4CAF50", "#F44336"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val}  ({val / len(df):.1%})",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "01_class_distribution.png")


def plot_completion_rate_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["avg_completion_rate"], bins=20, color="#2196F3", edgecolor="white", alpha=0.85)
    ax.axvline(0.40, color="#F44336", linewidth=2, linestyle="--", label="Threshold (0.40)")
    ax.set_title("avg_completion_rate Distribution")
    ax.set_xlabel("avg_completion_rate")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "02_completion_rate_dist.png")


def plot_numeric_distributions(df: pd.DataFrame, cols: list):
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])][:12]
    ncols = 3
    nrows = (len(cols) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=20, color="#9C27B0", alpha=0.7, edgecolor="white")
        axes[i].set_title(col, fontsize=9)
        axes[i].grid(alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "03_numeric_distributions.png")


def plot_features_vs_target(df: pd.DataFrame, features: list, target: str = "is_low_engagement"):
    plot_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])][:8]
    ncols = 2
    nrows = (len(plot_cols) + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(plot_cols):
        sns.boxplot(
            data=df, x=target, y=col,
            palette={"0": "#4CAF50", "1": "#F44336"},
            order=[0, 1],
            ax=axes[i],
        )
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("0 = High Engagement   1 = Low Engagement")
        axes[i].grid(axis="y", alpha=0.3)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Target Class", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "04_features_vs_target.png")


def plot_correlation_heatmap(df: pd.DataFrame, cols: list):
    numeric_df = df[cols].select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    _save(fig, "05_correlation_heatmap.png")


def plot_risk_by_media_type(df: pd.DataFrame, target: str = "is_low_engagement"):
    if "media_type" not in df.columns:
        return
    grp = df.groupby("media_type")[target].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(grp.index, grp.values, color="#FF9800", edgecolor="white")
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    ax.axhline(df[target].mean(), color="red", linestyle="--", linewidth=1.2, label="Overall average")
    ax.set_title("Low-Engagement Rate by Media Type")
    ax.set_ylabel("Fraction low engagement")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "06_risk_by_media_type.png")


def plot_risk_by_entry_source(df: pd.DataFrame, target: str = "is_low_engagement"):
    if "entry_source" not in df.columns:
        return
    grp = df.groupby("entry_source")[target].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grp.index, grp.values, color="#00BCD4", edgecolor="white")
    ax.axhline(df[target].mean(), color="red", linestyle="--", linewidth=1.2, label="Overall average")
    ax.set_title("Low-Engagement Rate by Entry Source")
    ax.set_ylabel("Fraction low engagement")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "07_risk_by_entry_source.png")


def run_eda(df: pd.DataFrame, feature_cols: list):
    print("Running EDA...")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicates: {df.duplicated().sum()}")
    print(f"  Low-engagement rate: {df['is_low_engagement'].mean():.1%}")

    plot_class_distribution(df)
    plot_completion_rate_distribution(df)
    plot_numeric_distributions(df, feature_cols)
    plot_features_vs_target(df, feature_cols)
    plot_correlation_heatmap(df, feature_cols)
    plot_risk_by_media_type(df)
    plot_risk_by_entry_source(df)
    print("EDA complete.\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_load import load_raw, drop_leakage_columns, build_target
    from features import engineer_features, get_feature_lists

    raw = load_raw()
    df = drop_leakage_columns(raw)
    df = build_target(df)
    df = engineer_features(df)
    num_cols, cat_cols = get_feature_lists(df)
    run_eda(df, num_cols + cat_cols)
