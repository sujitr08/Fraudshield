@echo off
setlocal
cd /d "%~dp0"
start "" wscript.exe "%~dp0Start_FraudShield.vbs"
endlocal
exit /b 0
