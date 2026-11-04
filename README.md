# Recovera AI (Version 3.0) — Complete Setup & Developer Guide

Recovera AI is a hospital-supervised post-discharge recovery management platform featuring clinical document ingestion via PaddleOCR, doctor review/approval workflows, dynamic recovery plan generation, medication scheduling and adherence tracking, daily symptom check-ins, conflict-free appointments, immediate 15-minute consultations, outbound SMS notifications via OnePlus 10T 5G (Android 15), and explainable recovery-risk prioritization.

---

## 💻 Running on Another Computer (Sharing with a Friend)

### ⚠️ Golden Rule: What to Copy vs What NOT to Copy
When sharing the project via ZIP, USB drive, or Git:

| Share These Files / Folders | DO NOT Copy (Delete before sharing) | Why? |
| :--- | :--- | :--- |
| `backend/` (code, scripts, schemas) | ❌ `backend/.venv` | Virtual environments contain machine-specific hardcoded paths. |
| `frontend/` (src, public, configs) | ❌ `frontend/node_modules` | Node packages are huge and platform-specific. |
| `docs/` & `tests/` | ❌ `frontend/dist` | Build artifacts will be generated freshly. |
| `docker-compose.yml` | ❌ `__pycache__` / `.pytest_cache` | Temporary Python bytecode. |
| `setup.bat` & `run.bat` | ❌ `storage/*` (keep empty folder) | Confidential patient uploads. |
| `README.md` | | |

---

## 🛠️ Prerequisites on the Target Computer

Before running, make sure the computer has:
1. **Python 3.11 or 3.12**: Download from [python.org](https://www.python.org/).  
   *(CRITICAL: Ensure the **"Add python.exe to PATH"** checkbox is ticked during installation!)*
2. **Node.js 18+ or 20+ LTS**: Download from [nodejs.org](https://nodejs.org/).
3. **Database (PostgreSQL 15+)**:
   - **Option A (Recommended & Easiest)**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.
   - **Option B (Native)**: Install [PostgreSQL](https://www.postgresql.org/download/) locally and create a database named `recovera` with user `recovera` and password `change-me`.

---

## ⚡ Option 1: Automated 1-Click Setup (Windows)

1. **Extract/Open** the project root folder.
2. If using Docker for PostgreSQL, start Docker Desktop and run:
   ```cmd
   docker compose up -d
   ```
3. Double-click **`setup.bat`**:
   - Automatically validates Python and Node.js.
   - Creates the Python virtual environment (`backend\.venv`).
   - Installs all backend packages from `backend\requirements.txt`.
   - Runs database migrations (`migrate_v3.py`) and seeds the default doctor account.
   - Runs `npm install` inside `frontend`.
4. Double-click **`run.bat`**:
   - Spawns the Backend API and the Frontend Web Portal in two separate windows.
   - Open your browser at: **[http://localhost:5173](http://localhost:5173)**.

---

## 🔧 Option 2: Step-by-Step Manual Setup

### 1. Database Setup
Using Docker (one-liner):
```cmd
docker compose up -d
```
*Or ensure your local PostgreSQL service is running on port `5432` with connection URL: `postgresql+psycopg://recovera:change-me@localhost:5432/recovera`.*

### 2. Backend Setup
Open a terminal in the project root:
```powershell
cd backend

# 1. Create environment file if missing
copy .env.example .env

# 2. Create virtual environment
python -m venv .venv

# 3. Upgrade pip and install dependencies
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt

# 4. Run schema migration & seed initial admin doctor
.\.venv\Scripts\python.exe scripts/migrate_v3.py
.\.venv\Scripts\python.exe scripts/create_admin.py
```

### 3. Frontend Setup
Open another terminal:
```powershell
cd frontend
npm.cmd install
```

---

## 🚀 Running the Servers Manually

Open **two separate terminals**:

### Terminal 1: Backend Server (FastAPI)
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- **API Root**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Terminal 2: Frontend Server (Vite / React)
```powershell
cd frontend
npm.cmd run dev
```
- **Web Portal**: [http://localhost:5173](http://localhost:5173)

---

## 🔑 Default Credentials & Role Testing

### 1. Doctor / Clinical Staff Portal
- **URL**: [http://localhost:5173](http://localhost:5173) (Select **"Staff Sign In"**)
- **Email**: `doctor@recovera.local`
- **Password**: `HospitalStaff@2026`
- **Staff Registration Secret** (if creating new staff): `RECOVERA-STAFF-2026`

### 2. Patient Portal
- Select **"Patient Sign In"** or register a new patient via **"Register as Patient"**.
- Patients can view their doctor-published recovery plan, listen with Multilingual TTS (English, Tamil, Hindi), track scheduled medication doses via the interactive Website Intake Tracker, request 15-minute consultations, and submit daily check-ins.

---

## 🧪 Running Automated Tests
Run all 21 automated system and contract test cases:
```powershell
.\backend\.venv\Scripts\python.exe -m pytest tests/ -v
```
Tests verify:
1. `tests/test_contract_phase0.py`: Zero WhatsApp rule, OnePlus 10T 5G SMS gateway, schema enums, bcrypt 72-byte safety.
2. `tests/test_phase1_auth_rbac.py`: JWT tokens, password hashing, RBAC role gates (Doctor, Nurse, Patient), registration secrets.
3. `tests/test_phase2_phase3.py`: Patient workspace, document upload, PaddleOCR extraction, approval workflows.
4. `tests/test_phase4_phase5.py`: Recovery plan generation & doctor approval, plan invalidation on document updates, multi-session medication schedules, exact occurrence calculation, historical adherence preservation.
