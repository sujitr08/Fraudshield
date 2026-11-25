"""Score a supported CSV outside Streamlit.

Example:
    python scripts/score_batch.py --input data/raw/financial_fraud_detection_dataset.csv \
        --model models/financial_transactions_5k_random_forest_class_weights.joblib \
        --output reports/scored_transactions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_core import get_feature_columns, normalize_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a supported fraud CSV with a saved pipeline.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input)
    normalized = normalize_dataset(raw)
    pipeline = joblib.load(args.model)
    probabilities = pipeline.predict_proba(normalized[get_feature_columns(normalized)])[:, 1]
    output = normalized.copy()
    output["risk_score"] = probabilities
    output["predicted_fraud"] = (probabilities >= args.threshold).astype(int)
    output["risk_band"] = pd.cut(probabilities, bins=[-0.001, min(args.threshold * 0.55, 0.49), args.threshold, 1.001], labels=["Low", "Medium", "High"]).astype(str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"saved {len(output):,} scored rows to {args.output}")
    print(f"alerts: {int(output['predicted_fraud'].sum()):,}")


if __name__ == "__main__":
    main()
