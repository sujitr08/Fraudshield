"""Download and validate the classic 284,807-row credit-card dataset.

The file is not bundled because the original dataset is distributed under its
source terms and is roughly 150 MB. Run this script when you have permission
to use it:

    python scripts/download_reference_dataset.py
"""

from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"


def validate(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"Time", "Amount", "Class"} | {f"V{index}" for index in range(1, 29)}
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError("Downloaded file does not match the expected Time/V1-V28/Amount/Class schema")
        rows = 0
        fraud = 0
        for row in reader:
            rows += 1
            fraud += int(row["Class"])
    if rows != 284807 or fraud != 492:
        raise ValueError(f"Validation failed: expected 284807 rows and 492 fraud cases, got {rows} rows and {fraud} fraud cases")
    print(f"validated reference dataset: {rows:,} rows, {fraud:,} fraud cases")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    urllib.request.urlretrieve(URL, OUTPUT)
    validate(OUTPUT)
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"download/validation failed: {exc}", file=sys.stderr)
        raise
