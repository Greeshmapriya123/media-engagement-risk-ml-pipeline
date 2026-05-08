import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Columns excluded from the feature matrix.
# IDs and free text carry no signal; target and its source columns must stay out.
DROP_COLS = [
    "object_id",
    "entry_name",
    "creator_name",
    "created_at",
    "is_low_engagement",
    "avg_completion_rate",
    "avg_view_drop_off",
]

CATEGORICAL_COLS = ["media_type", "entry_source", "status"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "duration_msecs" in df.columns:
        df["duration_mins"] = df["duration_msecs"] / 60_000

    if {"unique_viewers", "count_loads"}.issubset(df.columns):
        df["viewer_play_ratio"] = df["unique_viewers"] / (df["count_loads"] + 1)

    if {"unique_known_users", "unique_viewers"}.issubset(df.columns):
        df["known_user_share"] = df["unique_known_users"] / (df["unique_viewers"] + 1)

    if {"count_plays", "unique_viewers"}.issubset(df.columns):
        df["play_intensity"] = df["count_plays"] / (df["unique_viewers"] + 1)

    for col in ["count_plays", "sum_time_viewed", "count_loads", "unique_viewers"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df


def get_feature_lists(df: pd.DataFrame):
    exclude = set(DROP_COLS + CATEGORICAL_COLS)
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    return numeric_cols, cat_cols


def build_preprocessor(numeric_cols: list, cat_cols: list) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


if __name__ == "__main__":
    from data_load import load_raw, drop_leakage_columns, build_target
    raw = load_raw()
    df = drop_leakage_columns(raw)
    df = build_target(df)
    df = engineer_features(df)
    num, cat = get_feature_lists(df)
    print("Numeric features:", num)
    print("Categorical features:", cat)
