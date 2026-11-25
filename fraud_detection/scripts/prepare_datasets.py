"""Create canonical cleaned datasets and dataset summary artifacts.

Run from the project root:
    python scripts/prepare_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud_core import DATASETS, load_dataset  # noqa: E402


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def main() -> None:
    raw_dir = PROJECT_ROOT / "data" / "raw"
    cleaned_dir = PROJECT_ROOT / "data" / "cleaned"
    reports_dir = PROJECT_ROOT / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for dataset_name, config in DATASETS.items():
        source_path = Path(config["path"])
        raw_path = raw_dir / source_path.name
        if source_path.resolve() != raw_path.resolve():
            raw_path.write_bytes(source_path.read_bytes())

        cleaned = load_dataset(dataset_name)
        cleaned_path = cleaned_dir / f"{slugify(dataset_name)}_cleaned.csv"
        cleaned.to_csv(cleaned_path, index=False)
        summary_rows.append({
            "dataset": dataset_name,
            "source_file": source_path.name,
            "cleaned_file": cleaned_path.name,
            "rows": len(cleaned),
            "columns": len(cleaned.columns),
            "fraud_cases": int(cleaned["target"].sum()),
            "fraud_rate": float(cleaned["target"].mean()),
            "missing_values_after_cleaning": int(cleaned.isna().sum().sum()),
        })
        print(f"created {cleaned_path} ({len(cleaned):,} rows)")

    pd.DataFrame(summary_rows).to_csv(reports_dir / "dataset_summary.csv", index=False)
    print(f"created {reports_dir / 'dataset_summary.csv'}")


if __name__ == "__main__":
    main()
