@echo off
setlocal enabledelayedexpansion
title Recovera AI - Automated Setup Assistant

echo ============================================================
echo        RECOVERA AI (v3.0) - SYSTEM SETUP ASSISTANT
echo ============================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.11 or 3.12 from https://www.python.org/
    echo Make sure to check "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo [OK] Found %%i

:: 2. Check Node.js
node --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Node.js is not installed or not added to PATH.
    echo Please install Node.js 18+ or 20+ LTS from https://nodejs.org/
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('node --version') do echo [OK] Found Node.js %%i

:: 3. Prepare Environment Configuration (.env)
echo.
echo [1/5] Checking backend configuration (.env)...
if not exist "backend\.env" (
    echo [INFO] Creating backend\.env from .env.example...
    copy "backend\.env.example" "backend\.env" >nul
    echo [OK] backend\.env created successfully.
) else (
    echo [OK] backend\.env already exists.
)

:: 4. Setup Python Virtual Environment
echo.
echo [2/5] Setting up Python virtual environment (backend\.venv)...
if not exist "backend\.venv" (
    echo [INFO] Creating virtual environment at backend\.venv...
    python -m venv backend\.venv
)
echo [OK] Virtual environment ready.

:: 5. Install Backend Dependencies
echo.
echo [3/5] Installing backend Python packages from backend\requirements.txt...
backend\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
backend\.venv\Scripts\pip.exe install -r backend\requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)
echo [OK] Backend dependencies installed.

:: 6. Database Migration & Admin Seeding
echo.
echo [4/5] Running PostgreSQL migrations & admin user setup...
echo Note: Ensure PostgreSQL is running on port 5432 (e.g., via Docker: docker compose up -d)
backend\.venv\Scripts\python.exe backend\scripts\migrate_v3.py
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Database migration failed. Is PostgreSQL running on localhost:5432?
    echo If using Docker, run: docker compose up -d
    echo You can re-run migration later via: backend\.venv\Scripts\python.exe backend\scripts\migrate_v3.py
) else (
    backend\.venv\Scripts\python.exe backend\scripts\create_admin.py
    echo [OK] Database schema initialized and admin ready.
)

:: 7. Frontend Dependency Installation
echo.
echo [5/5] Installing frontend dependencies (npm install)...
cd frontend
call npm.cmd install
cd ..
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install frontend npm packages.
    pause
    exit /b 1
)
echo [OK] Frontend packages installed.

echo.
echo ============================================================
echo   RECOVERA AI SETUP COMPLETE!
echo ============================================================
echo To start both servers simultaneously, double-click: run.bat
echo Or start them manually:
echo   1. Backend:  backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
echo   2. Frontend: cd frontend ^&^& npm.cmd run dev
echo.
pause
