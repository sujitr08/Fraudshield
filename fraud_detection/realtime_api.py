"""FastAPI service for live transaction fraud scoring.

Run from the project root:
    python -m uvicorn realtime_api:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from src.realtime_service import DEFAULT_DB_PATH, DEFAULT_MODEL_PATH, init_db, load_pipeline, metrics, persist_event, read_alerts, score_payload


MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH", str(DEFAULT_MODEL_PATH)))
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", str(DEFAULT_DB_PATH)))
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

app = FastAPI(title="FraudShield Real-Time Scoring API", version="1.0.0")
pipeline = None


class TransactionEvent(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    customer_id: str = "LIVE-CUSTOMER"
    transaction_datetime: str | None = None
    amount: float = Field(..., ge=0)
    merchant_category: str = "Unknown"
    payment_method: str = "Unknown"
    device_type: str = "Unknown"
    location: str = "Unknown"
    is_international: int = Field(0, ge=0, le=1)
    previous_transactions: int = Field(0, ge=0)
    average_spend: float = Field(1.0, gt=0)
    account_age_days: int = Field(0, ge=0)
    suspicious_keyword: str = "No"


@app.on_event("startup")
def startup() -> None:
    global pipeline
    init_db(DB_PATH)
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
    pipeline = load_pipeline(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model_loaded": pipeline is not None, "threshold": THRESHOLD, "database": str(DB_PATH)}


@app.post("/transactions")
def score_transaction(event: TransactionEvent) -> dict[str, Any]:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Scoring model is not loaded")
    values = event.model_dump() if hasattr(event, "model_dump") else event.dict()
    result = score_payload(pipeline, values, THRESHOLD)
    persist_event(DB_PATH, result)
    return {key: value for key, value in result.items() if key != "payload"}


@app.get("/alerts")
def alerts(limit: int = Query(50, ge=1, le=500), high_only: bool = False) -> list[dict[str, Any]]:
    return read_alerts(DB_PATH, limit=limit, high_only=high_only)


@app.get("/metrics")
def service_metrics() -> dict[str, Any]:
    return metrics(DB_PATH)
