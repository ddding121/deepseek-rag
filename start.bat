@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment missing. Please follow README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.maxUploadSize 50
pause
