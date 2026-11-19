@echo off
title Recovera AI - GPU Backend & Free Cloudflare Tunnel
echo =========================================================================
echo   RECOVERA AI: UNLIMITED RTX 3050 GPU BACKEND ^& CLOUDFLARE PUBLIC TUNNEL
echo =========================================================================
echo.
echo [1/2] Starting GPU-Accelerated FastAPI Backend on port 8000...
start "Recovera AI - GPU Backend" cmd /k "cd /d "%~dp0backend" && ..\.venv-gpu\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

timeout /t 3 >nul

echo [2/2] Launching Free Cloudflare Public HTTPS Tunnel...
echo.
echo Look for your public https://*.trycloudflare.com link below!
echo You can use this link anywhere in the world with zero memory limits.
echo.
"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000
pause
