"""Train, evaluate, and persist fraud-detection models.

Examples:
    python scripts/train_models.py
    python scripts/train_models.py --dataset "Financial Transactions (5K)" --compare
    python scripts/train_models.py --model "Logistic Regression" --scaling RobustScaler --balance SMOTE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_core import BALANCE_OPTIONS, DATASETS, MODEL_NAMES, SCALING_OPTIONS, compare_models, load_dataset, train_model, tune_model  # noqa: E402


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud-detection models and save artifacts.")
    parser.add_argument("--dataset", default="Financial Transactions (5K)", choices=list(DATASETS))
    parser.add_argument("--model", default="Random Forest", choices=MODEL_NAMES)
    parser.add_argument("--scaling", default="StandardScaler", choices=SCALING_OPTIONS)
    parser.add_argument("--balance", default="Class weights", choices=BALANCE_OPTIONS)
    parser.add_argument("--tune", action="store_true", help="Run 3-fold hyperparameter tuning before saving.")
    parser.add_argument("--compare", action="store_true", help="Run the six-model benchmark and save it.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    df = load_dataset(args.dataset)

    if args.tune:
        bundle = tune_model(df, args.model, scaling=args.scaling, balance_method=args.balance)
    else:
        bundle = train_model(df, args.model, scaling=args.scaling, balance_method=args.balance)

    base_name = f"{slugify(args.dataset)}_{slugify(args.model)}_{slugify(args.balance)}"
    model_path = models_dir / f"{base_name}.joblib"
    metadata_path = models_dir / f"{base_name}_metadata.json"
    importance_path = reports_dir / f"{base_name}_feature_importance.csv"
    joblib.dump(bundle.pipeline, model_path)
    metadata = {
        "dataset": args.dataset,
        "model": args.model,
        "scaling": args.scaling,
        "balance_method": args.balance,
        "tuned": bundle.tuned,
        "best_params": bundle.best_params,
        "train_rows": bundle.train_rows,
        "test_rows": bundle.test_rows,
        "metrics": {key: float(bundle.metrics[key]) for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    bundle.feature_importance.to_csv(importance_path, index=False)
    print(f"saved model: {model_path}")
    print(f"saved metadata: {metadata_path}")
    print(f"ROC-AUC: {metadata['metrics']['roc_auc']:.4f}")

    if args.compare:
        comparison = compare_models(df, scaling=args.scaling, balance_method=args.balance)
        comparison_path = reports_dir / f"{slugify(args.dataset)}_model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print(f"saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
