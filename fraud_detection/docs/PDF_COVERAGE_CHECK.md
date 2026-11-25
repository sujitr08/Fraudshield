# PDF coverage verification

This checklist maps every point in the supplied Financial Fraud Detection Report to an implementation in this project.

| Requirement | Implementation | Status |
|---|---|---|
| Introduction and project objective | Overview and Project Notes in `app.py` | PASS |
| Dataset overview | `data/README.md`, Data Preparation page, and `reports/dataset_summary.csv` | PASS |
| 284,807 rows / 492 fraud reference dataset | `scripts/download_reference_dataset.py` downloads and validates the exact row and fraud counts | PASS |
| Missing-value handling | `SimpleImputer` in `src/fraud_core.py` | PASS |
| RobustScaler | Sidebar scaling control and pipeline | PASS |
| SMOTE | Sidebar balance control and imbalanced-learn pipeline | PASS |
| Random undersampling | Sidebar balance control and imbalanced-learn pipeline | PASS |
| PCA | `make_dimension_reduction(..., "PCA")` and Data Preparation page | PASS |
| t-SNE | `make_dimension_reduction(..., "t-SNE")` and Data Preparation page | PASS |
| Logistic Regression | Model Lab benchmark and tuning | PASS |
| KNN | Model Lab benchmark and tuning | PASS |
| SVC | Model Lab benchmark and tuning | PASS |
| Decision Tree | Model Lab benchmark and tuning | PASS |
| Random Forest | Default model, benchmark, saved artifact, and tuning | PASS |
| Accuracy | Holdout metrics | PASS |
| Precision | Holdout metrics | PASS |
| Recall | Holdout metrics | PASS |
| ROC-AUC | Holdout metrics and ROC curve | PASS |
| Confusion matrix | Model Lab | PASS |
| Interactive dashboard | Risk Dashboard | PASS |
| Real-time fraud detection | FastAPI `/transactions`, SQLite persistence, demo producer, and Real-time Operations page | PASS |
| LSTM future-work direction | Optional `src/lstm_extension.py` | PASS |
| Hyperparameter tuning | `tune_model()` with 3-fold GridSearchCV | PASS |
| Conclusion and future improvements | Project Notes and README | PASS |

The report's numeric ROC-AUC value is treated as a reported experiment result, not hardcoded into the application. The application calculates ROC-AUC from the selected dataset, preprocessing strategy, model, and holdout split.
