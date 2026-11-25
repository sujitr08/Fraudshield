"""Local real-time transaction scoring and SQLite alert persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .fraud_core import get_feature_columns, single_transaction


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "financial_transactions_5k_random_forest_class_weights.joblib"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "realtime_alerts.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS transaction_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                received_at TEXT NOT NULL,
                transaction_datetime TEXT,
                amount REAL NOT NULL,
                location TEXT,
                merchant_category TEXT,
                risk_score REAL NOT NULL,
                risk_band TEXT NOT NULL,
                predicted_fraud INTEGER NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Open',
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_events_received_at ON transaction_events(received_at DESC)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_events_risk_band ON transaction_events(risk_band)")
        connection.commit()


def load_pipeline(model_path: str | Path = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Build the canonical feature row expected by the trained pipeline."""
    row = single_transaction(
        amount=float(payload.get("amount", 0.0)),
        merchant_category=str(payload.get("merchant_category", "Unknown")),
        payment_method=str(payload.get("payment_method", "Unknown")),
        device_type=str(payload.get("device_type", "Unknown")),
        location=str(payload.get("location", "Unknown")),
        is_international=int(bool(payload.get("is_international", 0))),
        previous_transactions=int(payload.get("previous_transactions", 0)),
        average_spend=float(payload.get("average_spend", 1.0)),
        account_age_days=int(payload.get("account_age_days", 0)),
        suspicious_keyword=str(payload.get("suspicious_keyword", "No")),
    )
    if payload.get("transaction_id") is not None:
        row.loc[0, "transaction_id"] = str(payload["transaction_id"])
    if payload.get("customer_id") is not None:
        row.loc[0, "customer_id"] = str(payload["customer_id"])
    if payload.get("transaction_datetime") is not None:
        parsed = pd.to_datetime(payload["transaction_datetime"], errors="coerce")
        if not pd.isna(parsed):
            row.loc[0, "transaction_datetime"] = parsed
            row.loc[0, "transaction_hour"] = parsed.hour
            row.loc[0, "transaction_dayofweek"] = parsed.dayofweek
            row.loc[0, "transaction_seconds"] = parsed.hour * 3600 + parsed.minute * 60 + parsed.second
    return row


def score_payload(pipeline, payload: dict[str, Any], threshold: float = 0.5) -> dict[str, Any]:
    frame = payload_to_frame(payload)
    probability = float(pipeline.predict_proba(frame[get_feature_columns(frame)])[:, 1][0])
    band = "Low" if probability < min(threshold * 0.55, 0.49) else "Medium" if probability < threshold else "High"
    return {
        "transaction_id": str(frame.loc[0, "transaction_id"]),
        "customer_id": str(frame.loc[0, "customer_id"]),
        "transaction_datetime": pd.Timestamp(frame.loc[0, "transaction_datetime"]).isoformat(),
        "amount": float(frame.loc[0, "amount"]),
        "location": str(frame.loc[0, "location"]),
        "merchant_category": str(frame.loc[0, "merchant_category"]),
        "risk_score": probability,
        "risk_band": band,
        "predicted_fraud": int(probability >= threshold),
        "payload": payload,
    }


def persist_event(db_path: str | Path, result: dict[str, Any]) -> None:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO transaction_events
            (transaction_id, received_at, transaction_datetime, amount, location,
             merchant_category, risk_score, risk_band, predicted_fraud,
             review_status, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'Open', ?)
            """,
            (
                result["transaction_id"], utc_now(), result["transaction_datetime"], result["amount"],
                result["location"], result["merchant_category"], result["risk_score"], result["risk_band"],
                result["predicted_fraud"], json.dumps(result["payload"], default=str),
            ),
        )
        connection.commit()


def read_alerts(db_path: str | Path = DEFAULT_DB_PATH, limit: int = 50, high_only: bool = False) -> list[dict[str, Any]]:
    init_db(db_path)
    query = "SELECT id, transaction_id, received_at, transaction_datetime, amount, location, merchant_category, risk_score, risk_band, predicted_fraud, review_status FROM transaction_events"
    params: list[Any] = []
    if high_only:
        query += " WHERE risk_band = 'High'"
    query += " ORDER BY received_at DESC LIMIT ?"
    params.append(max(1, min(int(limit), 500)))
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(query, params).fetchall()]


def metrics(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        total, alerts, avg_score = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(predicted_fraud), 0), COALESCE(AVG(risk_score), 0) FROM transaction_events"
        ).fetchone()
        high = connection.execute("SELECT COUNT(*) FROM transaction_events WHERE risk_band = 'High'").fetchone()[0]
    return {"total_events": int(total), "alerts": int(alerts), "high_risk": int(high), "average_risk_score": float(avg_score)}
