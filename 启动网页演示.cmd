@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "HF_HUB_OFFLINE=1"
set "PYEXE=python"
if exist "%~dp0..\.venv\Scripts\python.exe" (
  set "PYEXE=%~dp0..\.venv\Scripts\python.exe"
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYEXE=%~dp0.venv\Scripts\python.exe"
)

"%PYEXE%" "%~dp0src\web_demo.py" %*
set "RESULT=%errorlevel%"
if not "%RESULT%"=="0" (
  echo Startup failed. Please send the error text above.
  pause
)
exit /b %RESULT%