import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "media_analytics.csv")

# engagement_ranking is dropped here because it is a composite rank derived
# from the final engagement outcome and has no place as a model input.
#
# avg_completion_rate and avg_view_drop_off are NOT dropped here because
# build_target() needs avg_completion_rate to create the label.
# Both are excluded from the feature matrix in features.py (DROP_COLS).
LEAKAGE_COLUMNS = ["engagement_ranking"]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def inspect(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("\nSample rows:")
    print(df.head(3).to_string())
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print(f"\nMissing values:\n{missing.to_string() if not missing.empty else '  none'}")
    print(f"Duplicates: {df.duplicated().sum()}")


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if cols:
        print(f"Dropping leakage columns: {cols}")
    return df.drop(columns=cols, errors="ignore")


def build_target(df: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """
    Creates the binary target column `is_low_engagement`.

    Content with avg_completion_rate below `threshold` is labelled 1 (at risk).
    The threshold is a business parameter set before any split, so it does not
    cause leakage.

    avg_completion_rate is excluded from the feature matrix inside features.py
    so the model cannot see it at prediction time.
    """
    if "avg_completion_rate" not in df.columns:
        raise ValueError("Column 'avg_completion_rate' not found in dataset.")
    df = df.copy()
    df["is_low_engagement"] = (df["avg_completion_rate"] < threshold).astype(int)
    rate = df["is_low_engagement"].mean()
    print(f"Target created — low-engagement rate: {rate:.1%}  (threshold={threshold})")
    return df


if __name__ == "__main__":
    raw = load_raw()
    inspect(raw)
    df = drop_leakage_columns(raw)
    df = build_target(df)
    print(df["is_low_engagement"].value_counts())
