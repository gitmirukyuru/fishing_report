@echo off
echo Stopping existing Streamlit processes...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501 " ^| findstr LISTENING') do (
    taskkill /PID %%p /F >nul 2>&1
)
timeout /t 1 /nobreak >nul
echo Starting dashboard on http://localhost:8501
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
