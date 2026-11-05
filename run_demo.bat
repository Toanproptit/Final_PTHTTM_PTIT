@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Chua co .venv. Hay xem README.md de cai dat.
    pause
    exit /b 1
)
echo Mo trinh duyet tai http://127.0.0.1:8000
echo Nhan Ctrl+C de dung server.
".venv\Scripts\python.exe" -m uvicorn diabetes_demo.api:app --host 127.0.0.1 --port 8000
pause
