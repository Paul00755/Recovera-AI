@echo off
title Recovera AI - Launcher

echo ============================================================
echo           LAUNCHING RECOVERA AI (VERSION 3.0)
echo ============================================================
echo.

:: 1. Optional Docker PostgreSQL Auto-start
where docker >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Docker detected. Ensuring PostgreSQL container is running...
    docker compose up -d
) else (
    echo [INFO] Docker not found in PATH; assuming native PostgreSQL is running on port 5432.
)

:: 2. Check if .venv exists
if not exist "backend\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first!
    pause
    exit /b 1
)

:: 3. Launch Backend in separate terminal
echo [1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Recovera AI - Backend API" cmd /k "cd backend && .venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"

:: 4. Launch Frontend in separate terminal
echo [2/2] Starting React/Vite Frontend on http://localhost:5173 ...
start "Recovera AI - Frontend UI" cmd /k "cd frontend && npm.cmd run dev"

echo.
echo ============================================================
echo   SERVERS STARTED SUCCESSFULLY!
echo ============================================================
echo.
echo   * Web Portal:       http://localhost:5173
echo   * Backend Docs:     http://127.0.0.1:8000/docs
echo   * System Health:    http://127.0.0.1:8000/api/health
echo.
echo Leave this window open or close it. The service windows will continue running.
echo ============================================================
pause
