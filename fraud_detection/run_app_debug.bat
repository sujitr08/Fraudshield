@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo             FraudShield launcher
echo ==================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer and try again.
    pause
    exit /b 1
)

python -c "import streamlit, plotly, pandas, numpy, sklearn, imblearn, joblib, fastapi, uvicorn, requests" >nul 2>&1
if errorlevel 1 (
    echo One or more project dependencies are missing. Installing requirements...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. Check your internet connection and Python environment.
        pause
        exit /b 1
    )
)

REM Reuse a healthy API if it is already running; otherwise start it in a new window.
python -c "import json, urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1)); raise SystemExit(0 if d.get('model_loaded') else 1)" >nul 2>&1
if errorlevel 1 (
    echo Starting the FraudShield real-time API on port 8000...
    start "FraudShield API" /D "%~dp0" cmd /k "python -m uvicorn realtime_api:app --host 127.0.0.1 --port 8000"
) else (
    echo A healthy FraudShield API is already running on port 8000. Reusing it.
)

echo.
echo Starting the Streamlit dashboard...
echo Keep the API terminal open while using Real-time Operations.
echo.
python -m streamlit run app.py --server.headless false
if errorlevel 1 (
    echo FraudShield exited with an error. Read the traceback above.
    pause
)
endlocal
