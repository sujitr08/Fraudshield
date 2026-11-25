"""Send real HTTP transaction events to the local scoring API."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send transactions to FraudShield's live API.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "cleaned" / "financial_transactions_5k_cleaned.csv")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input).head(args.count)
    url = args.api_url.rstrip("/") + "/transactions"
    for index, row in df.iterrows():
        payload = {
            "transaction_id": str(row["transaction_id"]), "customer_id": str(row.get("customer_id", "DEMO-CUSTOMER")),
            "transaction_datetime": str(row.get("transaction_datetime", "")), "amount": float(row["amount"]),
            "merchant_category": str(row.get("merchant_category", "Unknown")), "payment_method": str(row.get("payment_method", "Unknown")),
            "device_type": str(row.get("device_type", "Unknown")), "location": str(row.get("location", "Unknown")),
            "is_international": int(row.get("is_international", 0)), "previous_transactions": int(row.get("previous_transactions", 0)),
            "average_spend": max(float(row.get("average_spend", 1)), 1.0), "account_age_days": int(row.get("account_age_days", 0)),
            "suspicious_keyword": str(row.get("suspicious_keyword", "No")),
        }
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"{index + 1:03d} | {result['transaction_id']} | score={result['risk_score']:.3f} | {result['risk_band']}")
        time.sleep(max(args.interval, 0))


if __name__ == "__main__":
    main()
