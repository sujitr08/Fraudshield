from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"

DATASETS: dict[str, dict[str, Any]] = {
    "Financial Transactions (5K)": {"path": DATA_DIR / "financial_fraud_detection_dataset.csv", "target": "Fraudulent", "description": "Feature-rich payment transactions for fast interactive experimentation.", "profile": "financial"},
    "Synthetic Banking Transactions (50K)": {"path": DATA_DIR / "synthetic_fraud_dataset.csv", "target": "Fraud_Label", "description": "Synthetic banking activity with device, balance, and behavioural signals.", "profile": "synthetic"},
    "Credit Card Transactions (100K)": {"path": DATA_DIR / "credit_card_fraud_dataset.csv", "target": "IsFraud", "description": "Supplied credit-card sample for imbalance and model comparison demos.", "profile": "credit_card"},
}
REFERENCE_DATASET_PATH = DATA_DIR / "raw" / "creditcard.csv"
if REFERENCE_DATASET_PATH.exists():
    DATASETS["PDF Reference Credit Card Dataset (284K)"] = {"path": REFERENCE_DATASET_PATH, "target": "Class", "description": "The classic 284,807-row credit-card dataset described in the project report.", "profile": "creditcard_reference"}

BASE_NUMERIC_FEATURES = ["amount", "is_international", "previous_transactions", "average_spend", "account_age_days", "transaction_hour", "transaction_dayofweek", "transaction_seconds", "amount_to_avg_spend", "amount_to_balance", "suspicious_keyword_flag"]
CATEGORICAL_FEATURES = ["merchant_category", "payment_method", "device_type", "location", "suspicious_keyword"]
FEATURE_COLUMNS = BASE_NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_NAMES = ["Logistic Regression", "KNN", "SVC", "Decision Tree", "Random Forest", "Extra Trees"]
SCALING_OPTIONS = ["StandardScaler", "RobustScaler"]
BALANCE_OPTIONS = ["Class weights", "SMOTE", "Random undersampling"]


@dataclass
class ModelBundle:
    pipeline: ImbPipeline
    model_name: str
    metrics: dict[str, Any]
    feature_importance: pd.DataFrame
    train_rows: int
    test_rows: int
    scaling: str = "StandardScaler"
    balance_method: str = "Class weights"
    tuned: bool = False
    best_params: dict[str, Any] | None = None


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    extra = sorted([col for col in df.columns if col.startswith("v_")])
    return BASE_NUMERIC_FEATURES + extra + CATEGORICAL_FEATURES


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").astype(str).replace({"nan": "Unknown"})


def _parse_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def infer_profile(columns: list[str] | pd.Index) -> str:
    cols = set(columns)
    if {"Fraudulent", "Transaction_Amount", "Transaction_Date"}.issubset(cols):
        return "financial"
    if {"Fraud_Label", "Transaction_Amount", "Account_Balance"}.issubset(cols):
        return "synthetic"
    if {"IsFraud", "Amount", "TransactionDate"}.issubset(cols):
        return "credit_card"
    if {"Class", "Amount", "Time"}.issubset(cols) and any(str(c).startswith("V") for c in cols):
        return "creditcard_reference"
    raise ValueError("Unsupported CSV schema. Expected Fraudulent, Fraud_Label, IsFraud, or classic Class/V1-V28 credit-card columns.")


def normalize_dataset(raw: pd.DataFrame, profile: str | None = None) -> pd.DataFrame:
    """Convert supplied schemas, including classic creditcard.csv, to one modelling schema."""
    profile = profile or infer_profile(raw.columns)
    if profile == "financial":
        df = pd.DataFrame({"transaction_id": raw["Transaction_ID"], "customer_id": raw["Customer_ID"], "transaction_datetime": _parse_datetime(raw["Transaction_Date"]), "amount": raw["Transaction_Amount"], "merchant_category": raw["Merchant_Category"], "payment_method": raw["Payment_Method"], "device_type": raw["Device_Type"], "location": raw["Location"], "is_international": raw["Is_International"], "previous_transactions": raw["Previous_Transactions"], "average_spend": raw["Average_Spend"], "account_age_days": raw["Account_Age_Days"], "suspicious_keyword": raw["Suspicious_Keyword"], "target": raw["Fraudulent"]})
    elif profile == "synthetic":
        df = pd.DataFrame({"transaction_id": raw["Transaction_ID"], "customer_id": raw["User_ID"], "transaction_datetime": _parse_datetime(raw["Date"]), "amount": raw["Transaction_Amount"], "merchant_category": raw["Merchant_Category"], "payment_method": raw["Card_Type"], "device_type": raw["Device_Type"], "location": raw["Location"], "is_international": 0, "previous_transactions": raw["Daily_Transaction_Count"], "average_spend": raw["Account_Balance"].clip(lower=1), "account_age_days": raw["Card_Age"], "suspicious_keyword": np.where(raw["Previous_Fraudulent_Activity"] > 0, "Yes", "No"), "target": raw["Fraud_Label"]})
    elif profile == "credit_card":
        df = pd.DataFrame({"transaction_id": raw["TransactionID"], "customer_id": raw["MerchantID"].astype(str), "transaction_datetime": _parse_datetime(raw["TransactionDate"]), "amount": raw["Amount"], "merchant_category": raw["TransactionType"], "payment_method": raw["TransactionType"], "device_type": "Unknown", "location": raw["Location"], "is_international": 0, "previous_transactions": 0, "average_spend": raw["Amount"].median(), "account_age_days": 0, "suspicious_keyword": "No", "target": raw["IsFraud"]})
    elif profile == "creditcard_reference":
        seconds = pd.to_numeric(raw["Time"], errors="coerce").fillna(0)
        df = pd.DataFrame({"transaction_id": np.arange(1, len(raw) + 1), "customer_id": "Unknown", "transaction_datetime": pd.Timestamp("2023-01-01") + pd.to_timedelta(seconds, unit="s"), "amount": raw["Amount"], "merchant_category": "Credit Card", "payment_method": "Card", "device_type": "Unknown", "location": "Unknown", "is_international": 0, "previous_transactions": 0, "average_spend": raw["Amount"].median(), "account_age_days": 0, "suspicious_keyword": "No", "target": raw["Class"]})
        for col in raw.columns:
            if str(col).startswith("V"):
                df[f"v_{str(col)[1:].lower()}"] = raw[col].values
    else:
        raise ValueError(f"Unsupported dataset profile: {profile}")

    for col in BASE_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = _clean_text(df[col])
    df["transaction_datetime"] = df["transaction_datetime"].fillna(pd.Timestamp("2023-01-01"))
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    for col in ["amount", "average_spend", "account_age_days", "previous_transactions", "is_international"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["amount"] = df["amount"].clip(lower=0)
    df["average_spend"] = df["average_spend"].replace(0, np.nan).fillna(df["amount"].median()).clip(lower=1)
    df["account_age_days"] = df["account_age_days"].clip(lower=0)
    df["previous_transactions"] = df["previous_transactions"].clip(lower=0)
    df["is_international"] = df["is_international"].clip(0, 1)
    df["transaction_hour"] = df["transaction_datetime"].dt.hour
    df["transaction_dayofweek"] = df["transaction_datetime"].dt.dayofweek
    df["transaction_seconds"] = df["transaction_datetime"].dt.hour * 3600 + df["transaction_datetime"].dt.minute * 60 + df["transaction_datetime"].dt.second
    df["suspicious_keyword_flag"] = (df["suspicious_keyword"].str.lower() == "yes").astype(int)
    df["amount_to_avg_spend"] = (df["amount"] / df["average_spend"].replace(0, 1)).clip(0, 100)
    df["amount_to_balance"] = df["amount_to_avg_spend"]
    for col in [c for c in df.columns if c.startswith("v_")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.reset_index(drop=True)


def load_dataset(dataset_name: str) -> pd.DataFrame:
    config = DATASETS[dataset_name]
    return normalize_dataset(pd.read_csv(config["path"]), config["profile"])


def _model_estimator(model_name: str, balance_method: str, random_state: int = 42):
    weighted = "balanced" if balance_method == "Class weights" else None
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1200, class_weight=weighted, solver="liblinear", random_state=random_state)
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
    if model_name == "SVC":
        return SVC(kernel="rbf", probability=True, class_weight=weighted, random_state=random_state)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=14, min_samples_leaf=2, class_weight=weighted, random_state=random_state)
    if model_name == "Extra Trees":
        return ExtraTreesClassifier(n_estimators=220, max_depth=18, min_samples_leaf=2, class_weight=weighted, random_state=random_state, n_jobs=-1)
    return RandomForestClassifier(n_estimators=240, max_depth=16, min_samples_leaf=2, class_weight=weighted or "balanced_subsample", random_state=random_state, n_jobs=-1)


def build_pipeline(model_name: str = "Random Forest", scaling: str = "StandardScaler", balance_method: str = "Class weights") -> ImbPipeline:
    scaler = RobustScaler() if scaling == "RobustScaler" else StandardScaler()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", scaler)]), BASE_NUMERIC_FEATURES),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _one_hot_encoder())]), CATEGORICAL_FEATURES),
    ], remainder="passthrough", verbose_feature_names_out=True)
    sampler: Any = "passthrough"
    if balance_method == "SMOTE":
        sampler = SMOTE(random_state=42)
    elif balance_method == "Random undersampling":
        sampler = RandomUnderSampler(random_state=42)
    return ImbPipeline([("preprocessor", preprocessor), ("sampler", sampler), ("classifier", _model_estimator(model_name, balance_method))])


def evaluate_predictions(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, probabilities)
    return {"accuracy": accuracy_score(y_true, predictions), "precision": precision_score(y_true, predictions, zero_division=0), "recall": recall_score(y_true, predictions, zero_division=0), "f1": f1_score(y_true, predictions, zero_division=0), "roc_auc": roc_auc_score(y_true, probabilities), "average_precision": average_precision_score(y_true, probabilities), "confusion_matrix": cm, "y_true": np.asarray(y_true), "y_pred": np.asarray(predictions), "probabilities": np.asarray(probabilities), "fpr": fpr, "tpr": tpr, "roc_thresholds": roc_thresholds, "precision_curve": precision_curve, "recall_curve": recall_curve, "pr_thresholds": pr_thresholds, "classification_report": classification_report(y_true, predictions, target_names=["Legitimate", "Fraud"], output_dict=True, zero_division=0)}


def _fit_bundle(df: pd.DataFrame, pipeline: ImbPipeline, model_name: str, scaling: str, balance_method: str, tuned: bool = False, best_params: dict[str, Any] | None = None) -> ModelBundle:
    features = get_feature_columns(df)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df["target"].astype(int), test_size=0.25, random_state=42, stratify=df["target"])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return ModelBundle(pipeline, model_name, evaluate_predictions(y_test, predictions, probabilities), get_feature_importance(pipeline), len(X_train), len(X_test), scaling, balance_method, tuned, best_params)


def train_model(df: pd.DataFrame, model_name: str = "Random Forest", scaling: str = "StandardScaler", balance_method: str = "Class weights") -> ModelBundle:
    return _fit_bundle(df, build_pipeline(model_name, scaling, balance_method), model_name, scaling, balance_method)


def compare_models(df: pd.DataFrame, model_names: list[str] | None = None, scaling: str = "StandardScaler", balance_method: str = "Class weights", max_rows: int = 12000) -> pd.DataFrame:
    working = df.sample(max_rows, random_state=42) if len(df) > max_rows else df.copy()
    names = model_names or MODEL_NAMES
    features = get_feature_columns(working)
    X_train, X_test, y_train, y_test = train_test_split(working[features], working["target"], test_size=0.25, random_state=42, stratify=working["target"])
    rows = []
    for name in names:
        pipeline = build_pipeline(name, scaling, balance_method)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = evaluate_predictions(y_test, predictions, probabilities)
        rows.append({"Model": name, "Accuracy": metrics["accuracy"], "Precision": metrics["precision"], "Recall": metrics["recall"], "F1": metrics["f1"], "ROC-AUC": metrics["roc_auc"], "Average Precision": metrics["average_precision"]})
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def tune_model(df: pd.DataFrame, model_name: str, scaling: str = "StandardScaler", balance_method: str = "Class weights") -> ModelBundle:
    features = get_feature_columns(df)
    X_train, _, y_train, _ = train_test_split(df[features], df["target"], test_size=0.25, random_state=42, stratify=df["target"])
    pipeline = build_pipeline(model_name, scaling, balance_method)
    if model_name == "Logistic Regression":
        grid = {"classifier__C": [0.1, 1.0, 10.0]}
    elif model_name == "KNN":
        grid = {"classifier__n_neighbors": [3, 7, 11], "classifier__weights": ["uniform", "distance"]}
    elif model_name == "SVC":
        grid = {"classifier__C": [0.5, 1.0, 3.0], "classifier__gamma": ["scale", "auto"]}
    else:
        grid = {"classifier__max_depth": [8, 14, None], "classifier__min_samples_leaf": [1, 2]}
        if model_name in {"Random Forest", "Extra Trees"}:
            grid["classifier__n_estimators"] = [120, 220]
    search = GridSearchCV(pipeline, grid, scoring="average_precision", cv=3, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    return _fit_bundle(df, search.best_estimator_, model_name, scaling, balance_method, tuned=True, best_params=search.best_params_)


def get_feature_importance(pipeline: ImbPipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame({"feature": ["Model does not expose native importance"], "importance": [0.0]})
    importance = pd.DataFrame({"feature": names, "importance": values})
    importance["feature"] = importance["feature"].str.replace("numeric__", "", regex=False).str.replace("categorical__", "", regex=False)
    return importance.sort_values("importance", ascending=False).head(20).reset_index(drop=True)


def score_transactions(bundle: ModelBundle, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    scored = df.copy()
    probabilities = bundle.pipeline.predict_proba(scored[get_feature_columns(scored)])[:, 1]
    scored["risk_score"] = probabilities
    scored["predicted_fraud"] = (probabilities >= threshold).astype(int)
    scored["risk_band"] = pd.cut(probabilities, bins=[-0.001, min(threshold * 0.55, 0.49), threshold, 1.001], labels=["Low", "Medium", "High"]).astype(str)
    return scored


def make_dimension_reduction(df: pd.DataFrame, method: str = "PCA", max_points: int = 1200) -> pd.DataFrame:
    working = df.sample(max_points, random_state=42) if len(df) > max_points else df.copy()
    numeric = [c for c in get_feature_columns(working) if c in BASE_NUMERIC_FEATURES or c.startswith("v_")]
    matrix = SimpleImputer(strategy="median").fit_transform(working[numeric])
    matrix = RobustScaler().fit_transform(matrix)
    if method == "t-SNE":
        perplexity = min(30, max(5, (len(working) - 1) // 3))
        embedding = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(matrix)
    else:
        embedding = PCA(n_components=2, random_state=42).fit_transform(matrix)
    return pd.DataFrame({"component_1": embedding[:, 0], "component_2": embedding[:, 1], "label": working["target"].map({0: "Legitimate", 1: "Fraud"}).values})


def single_transaction(amount: float, merchant_category: str, payment_method: str, device_type: str, location: str, is_international: int, previous_transactions: int, average_spend: float, account_age_days: int, suspicious_keyword: str) -> pd.DataFrame:
    row = pd.DataFrame([{"transaction_id": "LIVE-REVIEW", "customer_id": "LIVE-REVIEW", "transaction_datetime": pd.Timestamp.now(), "amount": amount, "merchant_category": merchant_category, "payment_method": payment_method, "device_type": device_type, "location": location, "is_international": is_international, "previous_transactions": previous_transactions, "average_spend": average_spend, "account_age_days": account_age_days, "suspicious_keyword": suspicious_keyword, "target": 0}])
    row["transaction_hour"] = row["transaction_datetime"].dt.hour
    row["transaction_dayofweek"] = row["transaction_datetime"].dt.dayofweek
    row["transaction_seconds"] = row["transaction_datetime"].dt.hour * 3600 + row["transaction_datetime"].dt.minute * 60
    row["suspicious_keyword_flag"] = (row["suspicious_keyword"].str.lower() == "yes").astype(int)
    row["amount_to_avg_spend"] = (row["amount"] / row["average_spend"].replace(0, 1)).clip(0, 100)
    row["amount_to_balance"] = row["amount_to_avg_spend"]
    return row


def explain_transaction(row: pd.DataFrame, score: float, threshold: float) -> list[tuple[str, str]]:
    item = row.iloc[0]
    signals = []
    if item["amount_to_avg_spend"] >= 2: signals.append(("Amount anomaly", f"Amount is {item['amount_to_avg_spend']:.1f}× the stated average spend."))
    if item["is_international"] == 1: signals.append(("International activity", "Cross-border activity increases review priority."))
    if str(item["suspicious_keyword"]).lower() == "yes": signals.append(("Keyword signal", "The transaction contains a suspicious-keyword indicator."))
    if item["account_age_days"] < 90: signals.append(("New account", "The account is less than 90 days old."))
    if score >= threshold: signals.append(("Model decision", f"Risk score {score:.1%} is above the configured {threshold:.0%} threshold."))
    return signals or [("No dominant rule", "The model found no single high-signal rule; monitor the score with context.")]


def safe_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["transaction_id", "transaction_datetime", "amount", "merchant_category", "payment_method", "device_type", "location", "is_international", "risk_score", "predicted_fraud", "risk_band"]
    output = df[[col for col in columns if col in df.columns]].copy()
    if "transaction_datetime" in output: output["transaction_datetime"] = output["transaction_datetime"].dt.strftime("%d %b %Y %H:%M")
    if "risk_score" in output: output["risk_score"] = output["risk_score"].map(lambda value: f"{value:.1%}")
    if "amount" in output: output["amount"] = output["amount"].map(lambda value: f"₹{value:,.2f}")
    return output
