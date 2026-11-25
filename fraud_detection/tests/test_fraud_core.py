import pandas as pd

from src.fraud_core import FEATURE_COLUMNS, compare_models, load_dataset, make_dimension_reduction, normalize_dataset, train_model
from src.realtime_service import load_pipeline, metrics, persist_event, score_payload


def test_financial_dataset_normalizes_and_has_features():
    df = load_dataset("Financial Transactions (5K)")
    assert len(df) == 5000
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert df["target"].isin([0, 1]).all()


def test_model_trains_and_scores():
    df = load_dataset("Financial Transactions (5K)").sample(1200, random_state=7)
    bundle = train_model(df, "Logistic Regression")
    probabilities = bundle.pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    assert len(probabilities) == len(df)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_synthetic_schema_is_supported():
    raw = pd.DataFrame(
        {
            "Transaction_ID": ["TXN-1"],
            "User_ID": ["USER-1"],
            "Transaction_Amount": [10.0],
            "Transaction_Type": ["POS"],
            "Date": ["14 August 2023"],
            "Account_Balance": [1000.0],
            "Device_Type": ["Mobile"],
            "Location": ["Mumbai"],
            "Merchant_Category": ["Food"],
            "Previous_Fraudulent_Activity": [0],
            "Daily_Transaction_Count": [3],
            "Card_Type": ["Visa"],
            "Card_Age": [200],
            "Fraud_Label": [0],
        }
    )
    normalized = normalize_dataset(raw, "synthetic")
    assert normalized.loc[0, "target"] == 0
    assert normalized.loc[0, "merchant_category"] == "Food"


def test_classic_pdf_creditcard_schema_is_supported():
    raw = pd.DataFrame({"Time": [0, 60, 120, 180], "Amount": [10.0, 20.0, 30.0, 40.0], "Class": [0, 1, 0, 1]})
    for index in range(1, 29):
        raw[f"V{index}"] = 0.0
    normalized = normalize_dataset(raw)
    assert normalized.shape[0] == 4
    assert len([column for column in normalized.columns if column.startswith("v_")]) == 28


def test_full_model_comparison_and_pca_run():
    df = load_dataset("Financial Transactions (5K)").sample(400, random_state=4)
    comparison = compare_models(df, max_rows=400)
    assert set(comparison["Model"]) == {"Logistic Regression", "KNN", "SVC", "Decision Tree", "Random Forest", "Extra Trees"}
    embedding = make_dimension_reduction(df, "PCA", max_points=100)
    assert list(embedding.columns) == ["component_1", "component_2", "label"]


def test_realtime_scoring_and_persistence(tmp_path):
    pipeline = load_pipeline()
    result = score_payload(
        pipeline,
        {
            "transaction_id": "REALTIME-TEST-1",
            "amount": 250.0,
            "merchant_category": "Travel",
            "payment_method": "UPI",
            "device_type": "Mobile",
            "location": "Bengaluru",
            "is_international": 1,
            "previous_transactions": 2,
            "average_spend": 50.0,
            "account_age_days": 30,
            "suspicious_keyword": "Yes",
        },
        threshold=0.5,
    )
    db_path = tmp_path / "realtime.db"
    persist_event(db_path, result)
    service_metrics = metrics(db_path)
    assert result["transaction_id"] == "REALTIME-TEST-1"
    assert service_metrics["total_events"] == 1
