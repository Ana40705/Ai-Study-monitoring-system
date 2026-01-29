@echo off
title AI Student Monitor Console
echo ==================================================
echo      STARTING AI STUDENT MONITOR SYSTEM
echo ==================================================
echo.
echo [1/3] Activating AI Environment...
cd /d "%~dp0"
call venv\Scripts\activate

echo [2/3] Loading AI Models (YOLO + MediaPipe)...
echo       (This takes 10-15 seconds. Please wait...)

:: This command launches a separate timer that waits 12 seconds, 
:: then opens the browser automatically.
start /b cmd /c "timeout /t 12 /nobreak >nul && start http://localhost:5000"

echo [3/3] Starting Server...
echo.
:: Run the Python app
python app.py

pause