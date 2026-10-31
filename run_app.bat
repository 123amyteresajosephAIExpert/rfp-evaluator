@echo off
REM ---------------------------------------------------------------
REM  One-click launcher for Windows.
REM  First run: creates a private Python environment and installs
REM  the libraries (2-5 minutes). Later runs start in seconds.
REM ---------------------------------------------------------------
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating the project environment...
  python -m venv .venv
)
echo Installing / checking libraries...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
echo.
echo Starting the app. Your browser will open at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
