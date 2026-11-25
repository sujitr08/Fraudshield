"""Verify project artifacts and PDF requirement coverage locally."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fraud_core import get_feature_columns  # noqa: E402


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)
    print(f"PASS: {label}")


def main() -> None:
    required = [
        ROOT / "app.py", ROOT / "realtime_api.py", ROOT / "src" / "fraud_core.py",
        ROOT / "src" / "realtime_service.py", ROOT / "scripts" / "download_reference_dataset.py",
        ROOT / "data" / "cleaned" / "financial_transactions_5k_cleaned.csv",
        ROOT / "models" / "financial_transactions_5k_random_forest_class_weights.joblib",
        ROOT / "reports" / "financial_transactions_5k_model_comparison.csv",
    ]
    for path in required:
        check(f"artifact exists: {path.relative_to(ROOT)}", path.exists())

    for path in sorted((ROOT / "data" / "cleaned").glob("*.csv")):
        frame = pd.read_csv(path)
        check(f"cleaned data has rows: {path.name}", len(frame) > 0)
        check(f"cleaned data has target: {path.name}", "target" in frame.columns)
        check(f"cleaned data has no missing values: {path.name}", int(frame.isna().sum().sum()) == 0)

    model = joblib.load(ROOT / "models" / "financial_transactions_5k_random_forest_class_weights.joblib")
    frame = pd.read_csv(ROOT / "data" / "cleaned" / "financial_transactions_5k_cleaned.csv").head(3)
    probabilities = model.predict_proba(frame[get_feature_columns(frame)])[:, 1]
    check("saved model returns probabilities", len(probabilities) == 3 and ((probabilities >= 0) & (probabilities <= 1)).all())

    comparison = pd.read_csv(ROOT / "reports" / "financial_transactions_5k_model_comparison.csv")
    expected_models = {"Logistic Regression", "KNN", "SVC", "Decision Tree", "Random Forest", "Extra Trees"}
    check("comparison includes all six classifiers", set(comparison["Model"]) == expected_models)

    metadata = json.loads((ROOT / "models" / "financial_transactions_5k_random_forest_class_weights_metadata.json").read_text())
    check("evaluation metadata includes ROC-AUC", "roc_auc" in metadata["metrics"])
    check("real-time API exposes transaction endpoint", "/transactions" in (ROOT / "realtime_api.py").read_text())
    check("reference dataset validation is implemented", "284807" in (ROOT / "scripts" / "download_reference_dataset.py").read_text())
    print("PROJECT_VERIFICATION=PASS")


if __name__ == "__main__":
    main()
