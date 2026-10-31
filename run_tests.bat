@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run run_app.bat once first to set up the environment.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pytest -v
pause
