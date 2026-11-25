# FraudShield — Complete Streamlit Fraud Detection System

FraudShield is a full internship-ready fraud detection project built around the methodology described in the supplied Financial Fraud Detection Report. It combines preprocessing, imbalance treatment, dimensionality reduction, six supervised models, hyperparameter tuning, evaluation, live scoring, batch review, and a streaming simulator in one professional Streamlit application.

## PDF requirement coverage

| Report requirement | Implementation |
|---|---|
| Imbalanced fraud classification | Class distribution view, class-weighted models, SMOTE, and random undersampling |
| Missing-value handling | Median/mode imputation inside the modelling pipeline |
| Scaling | StandardScaler and RobustScaler selectable from the sidebar |
| PCA | Data Preparation page |
| t-SNE | Data Preparation page, capped sample for responsiveness |
| Logistic Regression | Model Lab comparison and tuning |
| KNN | Model Lab comparison and tuning |
| SVC | Model Lab comparison and tuning |
| Decision Tree | Model Lab comparison and tuning |
| Random Forest | Model Lab comparison and tuning; default model |
| Accuracy, precision, recall, F1, ROC-AUC | Model Lab metrics |
| Confusion matrix | Model Lab |
| Fraud-risk dashboard | Risk Dashboard |
| Real-time review | Live Scoring and Real-time Operations |
| Streaming concept | Streaming Monitor simulator plus live FastAPI event ingestion and SQLite alert persistence |
| Future work | Project Notes, including LSTM/autoencoder and Kafka/Spark roadmap |

## Important dataset note

The PDF describes the classic credit-card dataset with 284,807 rows and 492 fraud cases. That exact file was not included with the supplied attachments. The application supports its schema when a file named `creditcard.csv` or an equivalent CSV is uploaded through Batch Review:

```text
Time, V1, V2, ..., V28, Amount, Class
```

To obtain and validate the reference file in a permitted environment, run:

```bash
python scripts/download_reference_dataset.py
```

The script refuses to accept a file unless it contains exactly 284,807 rows and 492 fraud cases. After downloading, restart Streamlit and the reference dataset appears in the sidebar.

The project also includes the supplied 5K financial dataset, 50K synthetic banking dataset, and 100K credit-card sample so the app is immediately runnable.

## Application sections

- **Overview** — project brief, objectives, workflow, scope, and guardrails.
- **Risk Dashboard** — transaction volume, known fraud, risk distribution, exposure, location/category analysis, and review queue.
- **Data Preparation** — missing-value audit, class imbalance, scaler explanation, SMOTE/undersampling explanation, PCA, and t-SNE.
- **Live Scoring** — individual transaction scoring with risk score and rule-based review signals.
- **Batch Review** — filter, inspect, upload supported CSVs, and export alert queues.
- **Model Lab** — confusion matrix, ROC curve, classification report, feature importance, six-model comparison, and 3-fold hyperparameter tuning.
- **Streaming Monitor** — simulated event-by-event monitoring with threshold-based alerts.
- **Real-time Operations** — live FastAPI ingestion, model scoring, SQLite persistence, and an auto-refreshing alert queue.
- **Project Notes** — architecture, data contracts, production limitations, and future enhancements.

## Models and imbalance strategies

The sidebar controls the active scoring model, scaling method, and imbalance strategy:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Classifier
- Decision Tree
- Random Forest
- Extra Trees
- Class weights
- SMOTE
- Random undersampling
- StandardScaler
- RobustScaler

Hyperparameter tuning uses 3-fold cross-validation and average precision as the selection metric because accuracy alone is unreliable for imbalanced fraud detection.

The PDF's LSTM future-work direction is included as an optional extension in `src/lstm_extension.py`. Install `requirements-optional.txt` only if you want to experiment with sequence modelling; it is not required for the main Streamlit app.

## Run locally

### Windows (no PowerShell or Command Prompt window)

1. Extract the project folder.
2. Double-click `Start_FraudShield.vbs`. It starts the FastAPI service and Streamlit dashboard in the background and opens the browser when Streamlit is ready. The launcher uses `.venv\Scripts\pythonw.exe` when a project virtual environment exists; otherwise it uses `pythonw.exe` from PATH.
3. To stop the services, double-click `Stop_FraudShield.bat`.
4. Startup logs are saved in `logs/`. If the browser does not open, check `logs/launcher.log`, `logs/streamlit.log`, and `logs/realtime_api.log`.

`run_app.bat` also starts the quiet VBS launcher. For visible terminal output while troubleshooting, use `run_app_debug.bat`.

### Manual / cross-platform

```bash
cd fraud_detection_streamlit
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_app.py
```

The dashboard is available at `http://localhost:8501`; the API health check is `http://127.0.0.1:8000/health`.

Run tests:

```bash
python -m pytest -q
python scripts/verify_project.py
```

## Rebuild the data and model artifacts

The ZIP includes raw data, cleaned canonical data, a trained model, evaluation results, and a scored batch output. Recreate them from Python with:

```bash
python scripts/prepare_datasets.py
python scripts/train_models.py --compare
python scripts/score_batch.py \
  --input data/raw/financial_fraud_detection_dataset.csv \
  --model models/financial_transactions_5k_random_forest_class_weights.joblib \
  --output reports/financial_transactions_scored.csv
```

The launchers above start the real-time API automatically. To send sample transactions, open another terminal in the project folder and run:

```bash
python scripts/send_demo_stream.py --count 20 --interval 0.5
```

Then open the Streamlit **Real-time Operations** page. Each event is scored immediately by the saved model and persisted in `data/realtime_alerts.db`.

You can also train a different configuration:

```bash
python scripts/train_models.py \
  --model "Logistic Regression" \
  --scaling RobustScaler \
  --balance SMOTE \
  --tune
```

## Project structure

```text
fraud_detection_streamlit/
├── app.py
├── realtime_api.py               # Live REST scoring service
├── requirements.txt
├── .streamlit/config.toml
├── data/
│   ├── raw/                       # Original supplied CSV files
│   ├── cleaned/                   # Canonical cleaned CSV files
│   └── README.md
├── docs/Financial_Fraud_Detection_Report.pdf
├── models/                        # Saved joblib model + metadata
├── reports/                       # Dataset summary, benchmark, feature importance, scores
├── scripts/
│   ├── prepare_datasets.py
│   ├── train_models.py
│   ├── score_batch.py
│   ├── send_demo_stream.py
│   ├── download_reference_dataset.py
│   └── verify_project.py
├── src/fraud_core.py
├── src/lstm_extension.py
├── requirements-optional.txt
├── run_app.py
├── Start_FraudShield.vbs
├── Stop_FraudShield.bat
├── run_app.bat
├── run_app_debug.bat
└── tests/test_fraud_core.py
```

## Production limitations

This is an internship demonstration, not a production decision engine. Production use would require temporal validation, calibrated probabilities, cost-sensitive threshold approval, data-drift monitoring, access control, audit logs, privacy review, fairness validation, human review, and a real Kafka/Spark/database event source.


## Quiet Windows launcher

Double-click `Start_FraudShield.vbs` to launch without a terminal window. Use `Stop_FraudShield.bat` to stop the background services. Logs are in `logs/`. For visible troubleshooting output, use `run_app_debug.bat`.
