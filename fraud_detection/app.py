from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.fraud_core import (  # noqa: E402
    BALANCE_OPTIONS,
    DATASETS,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    SCALING_OPTIONS,
    compare_models,
    explain_transaction,
    get_feature_columns,
    load_dataset,
    make_dimension_reduction,
    normalize_dataset,
    safe_display_columns,
    score_transactions,
    single_transaction,
    tune_model,
    train_model,
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - gives a clearer local error
    px = None
    go = None


st.set_page_config(page_title="FraudShield | Fraud Detection System", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#11213d; --muted:#64748b; --navy:#102a43; --teal:#0f766e; --blue:#2563eb; --rose:#be123c; --soft:#f6f8fb; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(180deg,#f8fafc 0%,#ffffff 32%); color:var(--ink); }
    [data-testid="stSidebar"] { background: #0b1f3a; border-right: 1px solid #17375e; }
    [data-testid="stSidebar"] * { color: #e6effa !important; }
    [data-testid="stSidebar"] .stRadio label { padding: 7px 4px; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif; color:var(--ink); letter-spacing:-.02em; }
    .hero { padding: 28px 32px; border-radius: 24px; background: radial-gradient(circle at 85% 20%,#1e78a8 0,#173b67 35%,#0b1f3a 100%); color:#fff; margin-bottom:24px; box-shadow:0 12px 30px rgba(15,39,70,.14); }
    .hero h1 { color:#fff; font-size:2.4rem; margin:0 0 8px; }
    .hero p { color:#d8e7f7; max-width:780px; font-size:1.03rem; margin:0; line-height:1.6; }
    .eyebrow { color:#8dd7d0; text-transform:uppercase; letter-spacing:.14em; font-size:.74rem; font-weight:700; margin-bottom:8px; }
    .section-kicker { color:var(--teal); text-transform:uppercase; letter-spacing:.14em; font-size:.74rem; font-weight:700; margin:18px 0 4px; }
    .metric-card { background:#fff; border:1px solid #e5eaf0; border-radius:16px; padding:18px; box-shadow:0 3px 12px rgba(15,23,42,.04); min-height:112px; }
    .metric-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; font-weight:700; }
    .metric-value { color:var(--ink); font-family:'Space Grotesk',sans-serif; font-size:1.75rem; font-weight:700; margin-top:7px; }
    .metric-note { color:var(--muted); font-size:.78rem; margin-top:3px; }
    .callout { border-left:4px solid #0f766e; background:#ecfdf5; padding:14px 16px; border-radius:0 12px 12px 0; color:#164e63; line-height:1.55; }
    .warning { border-left-color:#eab308; background:#fefce8; color:#713f12; }
    .danger { border-left-color:#e11d48; background:#fff1f2; color:#881337; }
    .dark-card { background:#11213d; border-radius:16px; color:#e8f1fb; padding:18px 20px; }
    .dark-card h3 { color:#fff; margin-top:0; }
    .pill { display:inline-block; padding:5px 10px; border-radius:99px; font-size:.75rem; font-weight:700; background:#e0f2fe; color:#075985; }
    .pill.high { background:#ffe4e6; color:#9f1239; } .pill.medium { background:#fef3c7; color:#92400e; } .pill.low { background:#dcfce7; color:#166534; }
    .small { color:var(--muted); font-size:.88rem; line-height:1.55; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_dataset(dataset_name: str) -> pd.DataFrame:
    return load_dataset(dataset_name)


@st.cache_resource(show_spinner=False)
def get_model(dataset_name: str, model_name: str, scaling: str, balance_method: str):
    return train_model(get_dataset(dataset_name), model_name=model_name, scaling=scaling, balance_method=balance_method)


@st.cache_data(show_spinner=False)
def get_comparison(dataset_name: str, scaling: str, balance_method: str) -> pd.DataFrame:
    return compare_models(get_dataset(dataset_name), scaling=scaling, balance_method=balance_method)


@st.cache_resource(show_spinner=False)
def get_tuned_model(dataset_name: str, model_name: str, scaling: str, balance_method: str):
    return tune_model(get_dataset(dataset_name), model_name, scaling=scaling, balance_method=balance_method)


@st.cache_data(show_spinner=False)
def get_embedding(dataset_name: str, method: str) -> pd.DataFrame:
    return make_dimension_reduction(get_dataset(dataset_name), method=method)


def fmt_money(value: float) -> str:
    return f"₹{value:,.0f}"


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)


def plotly_chart(fig, key: str) -> None:
    fig.update_layout(template="plotly_white", font_family="DM Sans", margin=dict(l=8, r=8, t=42, b=8), height=340)
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_sidebar() -> tuple[str, str, str, float, str, str, str]:
    with st.sidebar:
        st.markdown("## 🛡️ FraudShield")
        st.caption("Financial fraud detection system")
        st.divider()
        page = st.radio("Workspace", ["Overview", "Risk Dashboard", "Data Preparation", "Live Scoring", "Batch Review", "Model Lab", "Streaming Monitor", "Real-time Operations", "Project Notes"], label_visibility="collapsed")
        st.divider()
        st.markdown("### Model controls")
        dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
        model_name = st.selectbox("Scoring model", MODEL_NAMES, index=4)
        scaling = st.selectbox("Scaling", SCALING_OPTIONS, index=0)
        balance_method = st.selectbox("Class balancing", BALANCE_OPTIONS, index=0)
        threshold_percent = st.slider("Fraud alert threshold", 10, 90, 50, 5, format="%d%%")
        threshold = threshold_percent / 100.0
        st.caption("Lower thresholds catch more suspicious activity but may increase false positives. SMOTE and undersampling run inside the training pipeline.")
        realtime_api_url = st.text_input("Real-time API URL", "http://127.0.0.1:8000")
        st.divider()
        st.markdown('<div class="small">Environment<br><b>Offline demo • v2.0</b><br>Model retrains automatically when controls change.</div>', unsafe_allow_html=True)
    return page, dataset_name, model_name, threshold, scaling, balance_method, realtime_api_url


page, dataset_name, model_name, threshold, scaling, balance_method, realtime_api_url = render_sidebar()
df = get_dataset(dataset_name)
bundle = get_model(dataset_name, model_name, scaling, balance_method)
scored = score_transactions(bundle, df, threshold)
fraud_rate = float(df["target"].mean())


def render_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="eyebrow">FraudShield / {page}</div><h1>{title}</h1><p class="small">{subtitle}</p>', unsafe_allow_html=True)


def render_overview() -> None:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Internship project • machine learning + analytics</div><h1>Fraud detection with an analyst-first workflow.</h1><p>FraudShield turns transaction data into explainable risk scores, review queues, and decision-ready monitoring views. It is designed as a practical demonstration of how a financial fraud detection system can move from raw data to a controlled, interactive application.</p></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]: metric_card("Transactions", f"{len(df):,}", "Current dataset")
    with cols[1]: metric_card("Known fraud", f"{int(df['target'].sum()):,}", f"{fraud_rate:.2%} of transactions")
    with cols[2]: metric_card("Features", str(len(get_feature_columns(df))), "Engineered + encoded")
    with cols[3]: metric_card("Validation ROC-AUC", f"{bundle.metrics['roc_auc']:.3f}", bundle.model_name)
    st.markdown('<div class="section-kicker">Project brief</div><h2>What this project demonstrates</h2>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("""
        <div class="small"><b>Fraud detection is a decision system, not only a classifier.</b><br><br>
        The application combines data preparation, behavioural feature engineering, class-imbalance-aware modelling, probability thresholding, explainable review signals, and operational monitoring. An investigator can score a single transaction, inspect a batch queue, study model performance, and download a review file without leaving the app.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="callout"><b>Recommended demo flow</b><br>
        1. Open <b>Risk Dashboard</b> to understand the portfolio.<br>
        2. Use <b>Live Scoring</b> to test an individual transaction.<br>
        3. Use <b>Batch Review</b> to filter and export alerts.<br>
        4. Use <b>Model Lab</b> to compare all six PDF models.<br>
        5. Use <b>Data Preparation</b> for SMOTE, scaling, PCA, and t-SNE.</div>
        """, unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">System flow</div><h2>From transaction to action</h2>', unsafe_allow_html=True)
    flow = st.columns(4)
    steps = [("01", "Prepare", "Parse dates, clean missing values, inspect imbalance, and choose RobustScaler or StandardScaler."), ("02", "Engineer", "Create amount ratios, time signals, categorical encodings, PCA, and t-SNE views."), ("03", "Compare", "Evaluate Logistic Regression, KNN, SVC, Decision Tree, Random Forest, and Extra Trees."), ("04", "Act", "Tune the selected model, simulate streaming alerts, and export a review worklist.")]
    for col, (num, title, desc) in zip(flow, steps):
        with col:
            st.markdown(f'<div class="dark-card"><div class="eyebrow">{num}</div><h3>{title}</h3><div class="small" style="color:#c9d9e8">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Scope and guardrails</div>', unsafe_allow_html=True)
    st.markdown('<div class="callout warning"><b>Important:</b> this is an internship demonstration using supplied datasets. A production deployment would require calibrated probabilities, drift monitoring, human review controls, access management, audit logging, privacy protections, and approval against a labelled validation sample.</div>', unsafe_allow_html=True)


def render_dashboard() -> None:
    render_header("Risk dashboard", "A portfolio-level view of transaction volume, fraud exposure, and model-prioritized review work.")
    high = scored[scored["risk_band"] == "High"]
    c = st.columns(4)
    with c[0]: metric_card("Transactions", f"{len(scored):,}", "Scored by current model")
    with c[1]: metric_card("High-risk queue", f"{len(high):,}", f"Threshold {threshold:.0%}")
    with c[2]: metric_card("Exposure in queue", fmt_money(float(high["amount"].sum())), "Amount requiring review")
    with c[3]: metric_card("Recall", f"{bundle.metrics['recall']:.1%}", "Holdout at 50% threshold")
    st.markdown('<div class="section-kicker">Monitoring views</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        trend = scored.assign(date=scored["transaction_datetime"].dt.date).groupby("date", as_index=False).agg(transactions=("target", "size"), known_fraud=("target", "sum"), flagged=("predicted_fraud", "sum"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["date"], y=trend["transactions"], mode="lines", name="Transactions", line=dict(color="#2563eb", width=2)))
        fig.add_trace(go.Scatter(x=trend["date"], y=trend["flagged"], mode="lines", name="Flagged", line=dict(color="#e11d48", width=2)))
        fig.update_layout(title="Daily volume vs. model flags", yaxis_title="Count", xaxis_title="Date", legend=dict(orientation="h", y=1.1))
        plotly_chart(fig, "daily-trend")
    with right:
        band_counts = scored["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
        band_counts.columns = ["Risk band", "Transactions"]
        fig = px.bar(band_counts, x="Risk band", y="Transactions", color="Risk band", color_discrete_map={"Low":"#16a34a", "Medium":"#d97706", "High":"#e11d48"}, title="Current risk distribution")
        plotly_chart(fig, "risk-distribution")
    left, right = st.columns(2)
    with left:
        cat = scored.groupby("merchant_category", as_index=False).agg(transactions=("target", "size"), known_fraud=("target", "sum"), flagged=("predicted_fraud", "sum")).sort_values("flagged", ascending=False)
        fig = px.bar(cat, x="merchant_category", y=["known_fraud", "flagged"], barmode="group", title="Fraud and model flags by category", labels={"value":"Count", "merchant_category":"Category", "variable":"Series"}, color_discrete_sequence=["#0f766e", "#e11d48"])
        plotly_chart(fig, "category-breakdown")
    with right:
        loc = scored.groupby("location", as_index=False).agg(transactions=("target", "size"), fraud_rate=("target", "mean"), flagged=("predicted_fraud", "sum")).sort_values("fraud_rate", ascending=False)
        loc["fraud_rate"] = loc["fraud_rate"] * 100
        fig = px.bar(loc, x="location", y="fraud_rate", color="fraud_rate", color_continuous_scale="Teal", title="Known fraud rate by location", labels={"fraud_rate":"Known fraud rate (%)", "location":"Location"})
        plotly_chart(fig, "location-rate")
    st.markdown('<div class="section-kicker">Priority queue preview</div>', unsafe_allow_html=True)
    queue = high.sort_values("risk_score", ascending=False).head(10)
    st.dataframe(safe_display_columns(queue), use_container_width=True, hide_index=True)


def render_data_preparation() -> None:
    render_header("Data preparation & feature engineering", "This section makes the PDF methodology visible: cleaning, imbalance treatment, scaling choices, and dimensionality reduction.")
    c = st.columns(4)
    with c[0]: metric_card("Rows", f"{len(df):,}", "Normalized dataset")
    with c[1]: metric_card("Fraud rate", f"{fraud_rate:.3%}", "Target prevalence")
    with c[2]: metric_card("Missing values", f"{int(df.isna().sum().sum()):,}", "After normalization")
    with c[3]: metric_card("Feature count", f"{len(get_feature_columns(df))}", "Before one-hot expansion")
    left, right = st.columns(2)
    with left:
        counts = df["target"].map({0: "Legitimate", 1: "Fraud"}).value_counts().rename_axis("Class").reset_index(name="Count")
        fig = px.bar(counts, x="Class", y="Count", color="Class", color_discrete_map={"Legitimate":"#2563eb", "Fraud":"#e11d48"}, title="Class imbalance before resampling")
        plotly_chart(fig, "class-imbalance")
    with right:
        nulls = df.isna().sum().sort_values(ascending=False).head(10).reset_index()
        nulls.columns = ["Feature", "Missing"]
        fig = px.bar(nulls, x="Feature", y="Missing", title="Missing-value audit", color="Missing", color_continuous_scale="Teal")
        plotly_chart(fig, "missing-audit")
    st.markdown('<div class="section-kicker">Preprocessing decisions</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="dark-card"><h3>Scaling</h3><div class="small" style="color:#c9d9e8"><b>StandardScaler</b> standardizes around the mean. <b>RobustScaler</b> uses median and IQR and is more resistant to extreme transaction amounts.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="dark-card"><h3>Balancing</h3><div class="small" style="color:#c9d9e8"><b>Class weights</b>, <b>SMOTE</b>, and <b>Random undersampling</b> are available inside the training pipeline. Compare them in Model Lab.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="dark-card"><h3>Leakage control</h3><div class="small" style="color:#c9d9e8">The train/test split happens before fitting preprocessing and resampling, so the holdout remains unseen during model fitting.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Dimensionality reduction</div><h2>PCA and t-SNE fraud separation</h2>', unsafe_allow_html=True)
    method = st.radio("Embedding", ["PCA", "t-SNE"], horizontal=True, help="t-SNE is computed on a capped sample for responsiveness.")
    if st.button(f"Generate {method} embedding", type="primary"):
        with st.spinner(f"Computing {method}..."):
            embedding = get_embedding(dataset_name, method)
        fig = px.scatter(embedding, x="component_1", y="component_2", color="label", color_discrete_map={"Legitimate":"#2563eb", "Fraud":"#e11d48"}, opacity=0.65, title=f"{method} projection of transaction features", labels={"component_1": f"{method} component 1", "component_2": f"{method} component 2"})
        plotly_chart(fig, f"embedding-{method}")
        st.caption("Projection is for visual exploration; it is not used as a substitute for supervised evaluation.")


def render_live_scoring() -> None:
    render_header("Live transaction scoring", "Simulate a real-time review: enter transaction attributes, receive a risk score, and inspect the strongest review signals.")
    st.markdown('<div class="callout">The score is a model probability estimate, not a final legal or operational decision. Use it to prioritize investigation and combine it with customer context.</div>', unsafe_allow_html=True)
    options = {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in ["merchant_category", "payment_method", "device_type", "location", "suspicious_keyword"]}
    with st.form("live_scoring_form"):
        a, b, c = st.columns(3)
        with a:
            amount = st.number_input("Transaction amount (₹)", min_value=0.0, value=float(df["amount"].median()), step=10.0)
            merchant_category = st.selectbox("Merchant category", options["merchant_category"])
            payment_method = st.selectbox("Payment method", options["payment_method"])
        with b:
            average_spend = st.number_input("Customer average spend (₹)", min_value=1.0, value=float(df["average_spend"].median()), step=10.0)
            previous_transactions = st.number_input("Previous transactions", min_value=0, value=int(df["previous_transactions"].median()), step=1)
            account_age_days = st.number_input("Account age (days)", min_value=0, value=int(df["account_age_days"].median()), step=1)
        with c:
            device_type = st.selectbox("Device type", options["device_type"])
            location = st.selectbox("Location", options["location"])
            is_international = st.toggle("International transaction", value=False)
            suspicious_keyword = st.selectbox("Suspicious keyword signal", options["suspicious_keyword"])
        submitted = st.form_submit_button("Score transaction", type="primary", use_container_width=True)
    if submitted:
        row = single_transaction(amount, merchant_category, payment_method, device_type, location, int(is_international), previous_transactions, average_spend, account_age_days, suspicious_keyword)
        score = float(bundle.pipeline.predict_proba(row[get_feature_columns(row)])[:, 1][0])
        label = "HIGH RISK" if score >= threshold else "REVIEW / LOW RISK"
        cls = "danger" if score >= threshold else "callout"
        st.markdown(f'<div class="{cls}"><b>{label}</b><br>Model risk score: <b>{score:.1%}</b> against the configured threshold of <b>{threshold:.0%}</b>.</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1: metric_card("Risk score", f"{score:.1%}", bundle.model_name)
        with m2: metric_card("Risk threshold", f"{threshold:.0%}", "Adjust in sidebar")
        with m3: metric_card("Amount / avg spend", f"{amount / max(average_spend, 1):.1f}×", "Behavioural signal")
        left, right = st.columns([1, 1.15])
        with left:
            if go:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=score * 100, number={"suffix":"%"}, title={"text":"Fraud risk score"}, gauge={"axis":{"range":[0,100]}, "bar":{"color":"#e11d48" if score >= threshold else "#0f766e"}, "steps":[{"range":[0,threshold*100],"color":"#dcfce7"},{"range":[threshold*100,100],"color":"#ffe4e6"}]}))
                fig.update_layout(height=280, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True, key="live-gauge")
        with right:
            st.markdown("#### Review signals")
            for title, detail in explain_transaction(row, score, threshold):
                st.markdown(f"**{title}**  \n<span class='small'>{detail}</span>", unsafe_allow_html=True)


def render_batch_review() -> None:
    render_header("Batch review queue", "Filter model-scored transactions into an investigator-ready worklist and export it for follow-up.")
    upload = st.file_uploader("Optional: score another supported CSV", type=["csv"], help="Supported schemas are documented in Project Notes.")
    working = scored.copy()
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
            working = score_transactions(bundle, normalize_dataset(raw), threshold)
            st.success(f"Uploaded file loaded: {len(working):,} rows.")
        except Exception as exc:
            st.error(str(exc))
    c1, c2, c3 = st.columns(3)
    with c1: risk_filter = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    with c2: location_filter = st.multiselect("Location", sorted(working["location"].unique()), default=sorted(working["location"].unique()))
    with c3: category_filter = st.multiselect("Merchant category", sorted(working["merchant_category"].unique()), default=sorted(working["merchant_category"].unique()))
    view = working[working["risk_band"].isin(risk_filter) & working["location"].isin(location_filter) & working["merchant_category"].isin(category_filter)].sort_values("risk_score", ascending=False)
    st.markdown(f"### {len(view):,} transactions in view")
    st.dataframe(safe_display_columns(view.head(500)), use_container_width=True, hide_index=True, height=470)
    export = view.copy()
    export["transaction_datetime"] = export["transaction_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.download_button("Download review queue (CSV)", export.to_csv(index=False).encode("utf-8"), "fraud_review_queue.csv", "text/csv", type="primary")


def render_model_lab() -> None:
    render_header("Model lab", "Explain how the current model performs on a held-out test set and which signals influence its decisions.")
    m = bundle.metrics
    c = st.columns(6)
    for col, label, value in zip(c, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg precision"], [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"], m["average_precision"]]):
        with col: metric_card(label, f"{value:.3f}", "Holdout evaluation")
    st.markdown(f'<div class="small">Model: <b>{bundle.model_name}</b> • Train rows: <b>{bundle.train_rows:,}</b> • Test rows: <b>{bundle.test_rows:,}</b> • The dashboard threshold can be changed independently for operational review.</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        cm = m["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Predicted legitimate", "Predicted fraud"], y=["Actual legitimate", "Actual fraud"], color_continuous_scale="Blues", title="Confusion matrix")
        plotly_chart(fig, "confusion-matrix")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m["fpr"], y=m["tpr"], mode="lines", name=f"ROC-AUC {m['roc_auc']:.3f}", line=dict(color="#2563eb", width=3)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline", line=dict(color="#94a3b8", dash="dash")))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        plotly_chart(fig, "roc-curve")
    left, right = st.columns([1.05, 1])
    with left:
        importance = bundle.feature_importance.head(12).sort_values("importance")
        fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Top model signals", color="importance", color_continuous_scale="Teal")
        plotly_chart(fig, "feature-importance")
    with right:
        report = pd.DataFrame(m["classification_report"]).T.reset_index().rename(columns={"index":"class"})
        report = report[report["class"].isin(["Legitimate", "Fraud"])][["class", "precision", "recall", "f1-score", "support"]]
        st.markdown("#### Classification report")
        st.dataframe(report.style.format({"precision":"{:.3f}", "recall":"{:.3f}", "f1-score":"{:.3f}", "support":"{:.0f}"}), use_container_width=True, hide_index=True)
        st.markdown('<div class="callout warning"><b>Metric interpretation:</b> in fraud detection, accuracy can be misleading when fraud is rare. Precision, recall, average precision, and the chosen operating threshold should be discussed together.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">PDF model comparison</div><h2>Six-model benchmark</h2>', unsafe_allow_html=True)
    st.caption("This benchmark evaluates the PDF-requested classifiers on the same stratified holdout. Larger datasets are capped at 12,000 rows for interactive responsiveness.")
    if st.button("Run full model comparison", type="primary"):
        with st.spinner("Training Logistic Regression, KNN, SVC, Decision Tree, Random Forest, and Extra Trees..."):
            comparison = get_comparison(dataset_name, scaling, balance_method)
        st.session_state["comparison"] = comparison
    if "comparison" in st.session_state:
        comparison = st.session_state["comparison"]
        st.dataframe(comparison.style.format({col: "{:.3f}" for col in comparison.columns if col != "Model"}), use_container_width=True, hide_index=True)
        fig = px.bar(comparison, x="Model", y="ROC-AUC", color="ROC-AUC", color_continuous_scale="Teal", title="ROC-AUC comparison")
        plotly_chart(fig, "model-comparison")
    st.markdown('<div class="section-kicker">Hyperparameter tuning</div><h2>Search for a stronger operating model</h2>', unsafe_allow_html=True)
    tune_name = st.selectbox("Model to tune", MODEL_NAMES, index=4, key="tune-model")
    if st.button("Run 3-fold hyperparameter tuning", type="secondary"):
        with st.spinner(f"Tuning {tune_name} with average precision as the selection metric..."):
            tuned = get_tuned_model(dataset_name, tune_name, scaling, balance_method)
        st.session_state["tuned_model"] = tuned
    if "tuned_model" in st.session_state:
        tuned = st.session_state["tuned_model"]
        tm = tuned.metrics
        tc = st.columns(5)
        for col, label, value in zip(tc, ["Precision", "Recall", "F1", "ROC-AUC", "Avg precision"], [tm["precision"], tm["recall"], tm["f1"], tm["roc_auc"], tm["average_precision"]]):
            with col: metric_card(f"Tuned {label}", f"{value:.3f}", tune_name)
        st.json(tuned.best_params or {})


def render_streaming_monitor() -> None:
    render_header("Streaming monitor", "A Kafka/Spark-inspired event simulator for demonstrating how a fraud score can become an operational alert.")
    st.markdown('<div class="callout warning"><b>Demo mode:</b> this page simulates an event stream from the supplied dataset. Production deployment would replace the iterator with Kafka, Spark Structured Streaming, or a database change feed.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: event_count = st.slider("Events to simulate", 10, 100, 30, 10)
    with c2:
        stream_threshold_percent = st.slider("Streaming threshold", 10, 90, int(round(float(threshold) * 100)), 5, format="%d%%")
        stream_threshold = stream_threshold_percent / 100.0
    with c3: st.metric("Configured model", bundle.model_name)
    if st.button("Start event stream", type="primary"):
        stream = scored.sort_values("transaction_datetime").head(event_count).copy()
        progress = st.progress(0, text="Connecting to simulated event source...")
        status = st.empty()
        alerts = []
        for index, (_, event) in enumerate(stream.iterrows(), start=1):
            is_alert = float(event["risk_score"]) >= stream_threshold
            if is_alert:
                alerts.append(event)
            status.markdown(f"**Event {index}/{event_count}** · `{event['transaction_id']}` · {event['location']} · ₹{event['amount']:,.2f} · {'🚨 ALERT' if is_alert else '✅ cleared'}")
            progress.progress(index / event_count, text=f"Processed {index}/{event_count} events")
            time.sleep(0.02)
        progress.empty()
        status.success(f"Stream complete: {len(alerts)} alert(s) generated from {event_count} events.")
        if alerts:
            alert_df = pd.DataFrame(alerts).sort_values("risk_score", ascending=False)
            st.markdown("#### Generated alert queue")
            st.dataframe(safe_display_columns(alert_df), use_container_width=True, hide_index=True)
            st.download_button("Download generated alerts", alert_df.to_csv(index=False).encode("utf-8"), "stream_alerts.csv", "text/csv")
        else:
            st.info("No events crossed the configured threshold in this simulation.")


def render_realtime_operations() -> None:
    render_header("Real-time operations", "Monitor transactions scored by the live FastAPI service. This is a real HTTP ingestion path backed by SQLite, not a replay-only chart.")
    st.markdown('<div class="callout">Start the API in a second terminal, then send events with the included demo producer. This page polls the API every three seconds and displays the persisted alert queue.</div>', unsafe_allow_html=True)
    st.code("python -m uvicorn realtime_api:app --host 127.0.0.1 --port 8000\npython scripts/send_demo_stream.py --count 20 --interval 0.5", language="bash")

    def live_panel() -> None:
        try:
            health_response = requests.get(f"{realtime_api_url.rstrip('/')}/health", timeout=2)
            health_response.raise_for_status()
            health = health_response.json()
            metrics_response = requests.get(f"{realtime_api_url.rstrip('/')}/metrics", timeout=2)
            metrics_response.raise_for_status()
            live_metrics = metrics_response.json()
            alerts_response = requests.get(f"{realtime_api_url.rstrip('/')}/alerts?limit=50", timeout=2)
            alerts_response.raise_for_status()
            alerts_data = alerts_response.json()
        except requests.RequestException:
            st.error("Real-time API is offline. Start `realtime_api.py` with the command above.")
            return

        c = st.columns(4)
        with c[0]: metric_card("API status", "ONLINE", f"Model loaded: {health.get('model_loaded', False)}")
        with c[1]: metric_card("Events received", f"{live_metrics.get('total_events', 0):,}", "SQLite event store")
        with c[2]: metric_card("High-risk alerts", f"{live_metrics.get('high_risk', 0):,}", "Configured threshold")
        with c[3]: metric_card("Average risk", f"{live_metrics.get('average_risk_score', 0):.1%}", "All received events")
        if alerts_data:
            alert_frame = pd.DataFrame(alerts_data)
            alert_frame["risk_score"] = alert_frame["risk_score"].map(lambda value: f"{value:.1%}")
            alert_frame["received_at"] = pd.to_datetime(alert_frame["received_at"], errors="coerce").dt.strftime("%d %b %H:%M:%S")
            st.dataframe(alert_frame, use_container_width=True, hide_index=True)
        else:
            st.info("The API is online but no transactions have been received yet.")

    if hasattr(st, "fragment"):
        live_panel = st.fragment(run_every="3s")(live_panel)
    live_panel()


def render_notes() -> None:
    render_header("Project notes", "A concise technical handoff for an internship review, viva, or project demonstration.")
    st.markdown('<div class="section-kicker">Architecture</div><h2>Designed as a small production-shaped system</h2>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("""
        **Data layer**  
        Three supplied CSV schemas are normalized into one canonical transaction model. The classic PDF `creditcard.csv` contract (`Time`, `V1`–`V28`, `Amount`, `Class`) is also recognized when uploaded.

        **Model layer**  
        The pipeline supports RobustScaler, StandardScaler, class weights, SMOTE, random undersampling, six classifiers, and 3-fold hyperparameter tuning selected by average precision.
        """)
    with b:
        st.markdown("""
        **Application layer**  
        Streamlit provides navigation, live scoring, batch review, monitoring charts, PCA/t-SNE exploration, streaming simulation, evaluation views, and CSV export.

        **Governance layer**  
        Thresholding, review signals, holdout metrics, and limitations are visible in the interface so a model score is not presented as an unexplained decision.
        """)
    st.markdown('<div class="section-kicker">Supported data contracts</div>', unsafe_allow_html=True)
    contracts = pd.DataFrame([
        ["PDF reference credit card", "Class", "Time, V1–V28, Amount, Class; expected report scale: 284,807 rows / 492 fraud"],
        ["Financial", "Fraudulent", "Transaction_Amount, Merchant_Category, Payment_Method, ..."],
        ["Synthetic banking", "Fraud_Label", "Transaction_Amount, Account_Balance, Device_Type, ..."],
        ["Supplied credit card", "IsFraud", "Amount, TransactionDate, MerchantID, TransactionType, ..."],
    ], columns=["Profile", "Target column", "Required signal family"])
    st.dataframe(contracts, use_container_width=True, hide_index=True)
    st.markdown('<div class="section-kicker">Future enhancements</div>', unsafe_allow_html=True)
    st.markdown("""
    - Add LSTM/autoencoder experiments as optional deep-learning extensions.
    - Calibrate probabilities and tune the operating threshold against an explicit cost matrix.
    - Replace the streaming simulator with Kafka or Spark Structured Streaming.
    - Add SHAP or native local explanations after validating explanation fidelity.
    - Add model drift, data quality, audit logs, role-based access, and notification integrations.
    - Validate fairness and performance across customer segments before production use.
    """)
    st.markdown('<div class="callout danger"><b>Production caution:</b> never auto-decline a customer solely from this demo score. Fraud operations should combine model output with policy rules, analyst review, and documented escalation paths.</div>', unsafe_allow_html=True)


if page == "Overview":
    render_overview()
elif page == "Risk Dashboard":
    render_dashboard()
elif page == "Data Preparation":
    render_data_preparation()
elif page == "Live Scoring":
    render_live_scoring()
elif page == "Batch Review":
    render_batch_review()
elif page == "Model Lab":
    render_model_lab()
elif page == "Streaming Monitor":
    render_streaming_monitor()
elif page == "Real-time Operations":
    render_realtime_operations()
else:
    render_notes()
