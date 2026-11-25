# FraudShield fixes applied

- Disabled Streamlit headless mode so the dashboard can open a browser automatically.
- Fixed the fraud-threshold sliders to display integer percentages (10–90%) and convert to decimal probabilities internally.
- Updated the Windows launcher to check dependencies, start the FastAPI service in a separate terminal, reuse an already healthy API, and then launch Streamlit.
- Updated the cross-platform Python launcher to start the API and dashboard together, wait briefly for API readiness, and stop the API process when the dashboard exits.
- Updated README instructions and added a clearer API health-check URL.

## Verification performed

- Python compilation passed for the application, API, service, and demo producer.
- Pytest: 6 tests passed.
- Project verification script passed.
- Saved-model probability inference passed.
- FastAPI startup and model loading passed.
- HTTP health check returned `model_loaded: true`.
- A test transaction was accepted by `/transactions`, scored, persisted in SQLite, and reflected by `/metrics`.

## Local browser test

The repair environment did not have Streamlit installed, so a browser-based Streamlit smoke test could not be run here. On Windows, launch with `run_app.bat`; it now starts the API and dashboard together. Keep the API terminal open during the demo.


## Quiet Windows launch (v3)

- Added `Start_FraudShield.vbs`, which starts the Python supervisor using `pythonw.exe` so no PowerShell/Command Prompt console stays open.
- The launcher starts FastAPI and Streamlit as hidden background processes and logs their output under `logs/`.
- Added `Stop_FraudShield.bat` to stop the background services.
- `run_app.bat` now opens the quiet VBS launcher; `run_app_debug.bat` retains the visible console launcher for troubleshooting.
- The launcher opens the browser only after Streamlit's health endpoint responds.
