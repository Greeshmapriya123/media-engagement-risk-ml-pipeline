"""
Compares the training data distribution against production requests
logged by the API to detect data drift over time.

Usage
-----
1. Start the API and score some items so prediction_log.jsonl accumulates.
2. Run this script: python monitoring/drift_report.py
3. Open reports/drift_report.html in a browser.
"""

import os
import sys
import json
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = os.path.join(os.path.dirname(__file__), "..")
LOG_FILE  = os.path.join(ROOT, "monitoring", "prediction_log.jsonl")
REPORT_OUT = os.path.join(ROOT, "reports", "drift_report.html")
sys.path.insert(0, os.path.join(ROOT, "src"))


def load_reference() -> pd.DataFrame:
    from data_load import load_raw, drop_leakage_columns, build_target
    from features import engineer_features, get_feature_lists

    raw = load_raw()
    df  = drop_leakage_columns(raw)
    df  = build_target(df)
    df  = engineer_features(df)
    num_cols, cat_cols = get_feature_lists(df)
    return df[num_cols + cat_cols + ["is_low_engagement"]]


def load_production() -> pd.DataFrame:
    if not os.path.exists(LOG_FILE):
        print(f"Log file not found: {LOG_FILE}")
        return pd.DataFrame()

    rows = []
    with open(LOG_FILE) as f:
        for line in f:
            entry = json.loads(line)
            row = entry.get("input_features", {})
            row["predicted_class"] = entry.get("predicted_class")
            rows.append(row)

    if not rows:
        print("Log file is empty.")
        return pd.DataFrame()

    return pd.DataFrame(rows)


def run_drift_report():
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError:
        print("Evidently not installed. Run: pip install evidently")
        return

    reference  = load_reference()
    production = load_production()

    if production.empty:
        print("No production data to compare against.")
        return

    common_cols = [c for c in reference.columns if c in production.columns]
    if not common_cols:
        print("No overlapping columns between reference and production data.")
        return

    print(f"Comparing {len(reference)} reference rows vs {len(production)} production rows "
          f"on {len(common_cols)} features...")

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference[common_cols], current_data=production[common_cols])

    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
    report.save_html(REPORT_OUT)
    print(f"Report saved: {REPORT_OUT}")


def check_prediction_rate():
    """Quick check: has the fraction of at-risk predictions shifted?"""
    production = load_production()
    if production.empty or "predicted_class" not in production.columns:
        return
    rate = production["predicted_class"].mean()
    print(f"Production at-risk rate: {rate:.1%}")
    print("Compare this against the training rate (~39%) to spot target drift.")


# Recommended monitoring thresholds (for reference when interpreting the report):
#
#   Jensen-Shannon distance per feature   > 0.10  → investigate
#   Fraction of features drifted          > 20%   → retrain candidate
#   Shift in mean predicted probability   > 5 pp  → investigate
#   Shift in actual at-risk rate          > 5 pp  → retrain candidate


if __name__ == "__main__":
    check_prediction_rate()
    run_drift_report()
