@echo off
setlocal
cd /d "%~dp0"
set "PIDFILE=%~dp0runtime_pids.json"
if not exist "%PIDFILE%" (
    echo FraudShield is not running, or its launcher has already stopped.
    timeout /t 2 >nul
    exit /b 0
)

REM Stop the Streamlit child first; the hidden launcher then shuts down its API child.
for /f "usebackq tokens=2 delims=:," %%P in (`findstr /i "streamlit_pid" "%PIDFILE%"`) do (
    set "STREAMLIT_PID=%%P"
)
for /f "usebackq tokens=2 delims=:," %%P in (`findstr /i "api_pid" "%PIDFILE%"`) do (
    set "API_PID=%%P"
)
set "STREAMLIT_PID=%STREAMLIT_PID: =%"
set "API_PID=%API_PID: =%"
if defined STREAMLIT_PID if not "%STREAMLIT_PID%"=="0" taskkill /PID %STREAMLIT_PID% /T /F >nul 2>&1
if defined API_PID if not "%API_PID%"=="0" taskkill /PID %API_PID% /T /F >nul 2>&1
del "%PIDFILE%" >nul 2>&1
echo FraudShield services stopped.
timeout /t 2 >nul
endlocal
