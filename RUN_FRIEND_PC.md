# Recovera AI — Setup & Run Guide (Friend's PC / Git Clone)

Welcome to **Recovera AI**! This guide contains step-by-step instructions to clone, install, configure, and run the project on a fresh computer (Windows, macOS, or Linux).

---

## 📋 1. Prerequisites (Install These First)

Ensure the following tools are installed on your computer before starting:

1. **Git**: [Download Git](https://git-scm.com/downloads)
2. **Python (v3.11 or v3.12)**: [Download Python](https://www.python.org/downloads/)
   > [!IMPORTANT]
   > On Windows, during Python installation, **you MUST check the box**:  
   > ☑️ **"Add python.exe to PATH"** (at the bottom of the first installer screen).
3. **Node.js (v18.x or v20.x LTS)**: [Download Node.js](https://nodejs.org/) (includes `npm`).

### Verify Installations in Terminal
Open your terminal (PowerShell, Command Prompt, or Bash) and run:
```bash
python --version
node --version
npm --version
git --version
```
*(All commands should return version numbers without error).*

---

## 📥 2. Clone the Repository

Open your terminal and clone the repository:

```bash
git clone <YOUR_GIT_REPOSITORY_URL>
cd "Recovera AI"
```
*(Replace `<YOUR_GIT_REPOSITORY_URL>` with your actual Git repo link).*

---

## ⚡ 3. Quick Setup (Windows Users Only)

If you are on Windows, automated setup scripts are included in the repository:

1. **Run Automated Setup**:
   Double-click **`setup.bat`** (or execute `.\setup.bat` in Command Prompt / PowerShell).  
   This will automatically:
   * Verify your Python and Node.js installations.
   * Create the backend Python virtual environment (`backend\.venv`).
   * Install all backend dependencies from `backend\requirements.txt`.
   * Create the `backend\.env` file from `.env.example`.
   * Initialize the SQLite database and bootstrap the default Admin account.
   * Install frontend dependencies with `npm install`.

2. **Launch the Application**:
   Double-click **`run.bat`** (or execute `.\run.bat`).  
   This starts both the Backend API (`http://127.0.0.1:8000`) and Frontend (`http://localhost:5173`) in two separate windows.

---

## 🛠️ 4. Manual Setup (Cross-Platform: Windows, macOS, Linux)

If you are on macOS/Linux or prefer setting up manually step-by-step, follow the instructions below:

### Part A: Backend Setup

Open a terminal in the root `Recovera AI` folder:

```bash
# 1. Navigate into the backend directory
cd backend

# 2. Create your environment configuration file
# On Windows (cmd):
copy .env.example .env
# On Windows (PowerShell) / macOS / Linux:
cp .env.example .env

# 3. Create a Python virtual environment
python -m venv .venv

# 4. Activate the virtual environment:
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Windows (CMD):
.\.venv\Scripts\activate.bat
# On macOS / Linux:
source .venv/bin/activate

# 5. Upgrade pip and install all required Python packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# 6. Initialize database tables and create the default Admin account
python scripts/create_admin.py
```

> [!NOTE]
> **Database Zero-Config:**
> By default, Recovera AI automatically creates and uses a local SQLite database (`recovera.db`). You do **not** need to install PostgreSQL or Docker to run the app. If you do wish to use PostgreSQL, start it via `docker compose up -d` and the backend will automatically connect.

#### Backend `.env` Configuration
Open `backend/.env` in any text editor and review these settings:
* **`GEMINI_API_KEY=`**: (Optional) Paste your free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/) for AI chat & multilingual support. If left empty, the app automatically uses the built-in clinical rule-based fallback AI.
* **`ONEPLUS_GATEWAY_ENABLED=false`**: Keep this set to `false` unless you are connected to the physical OnePlus SMS companion hardware. In simulation mode (`false`), SMS jobs are safely queued and simulated in logs without requiring any mobile device.
* **`STAFF_REGISTRATION_CODE=RECOVERA-STAFF-2026`**: Authorization code required when registering as a Doctor or Nurse.

---

### Part B: Frontend Setup

Open a **new terminal window** in the root `Recovera AI` folder:

```bash
# 1. Navigate into the frontend directory
cd frontend

# 2. Ensure frontend .env exists with API URL
# On Windows (cmd):
copy .env.example .env
# On Windows (PowerShell) / macOS / Linux:
cp .env.example .env

# 3. Install frontend npm dependencies
npm install
```

---

## 🚀 5. Running the Application

Always run both the Backend and Frontend servers simultaneously:

### Terminal 1: Backend API
```bash
cd backend

# Activate virtual environment:
# Windows (PowerShell): .\.venv\Scripts\Activate.ps1
# Windows (CMD):        .\.venv\Scripts\activate.bat
# macOS/Linux:          source .venv/bin/activate

# Start FastAPI server:
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
* **API Server**: `http://127.0.0.1:8000`
* **Swagger API Documentation**: `http://127.0.0.1:8000/docs`
* **Health Check**: `http://127.0.0.1:8000/api/health`

### Terminal 2: Frontend UI
```bash
cd frontend

# Start Vite dev server:
npm run dev
```
* **Web Portal**: `http://localhost:5173`

---

## 🔑 6. Logging In & First-Time Use

Open your web browser and go to: **`http://localhost:5173`**

### Pre-Configured Administrator Account
* **Email**: `admin@hospital.org`
* **Password**: `AdminSecure2026!`
* *(Use this account to view the Admin Dashboard, audit logs, system roster, and analytics).*

### How to Create or Reset an Admin Account via CLI
If you want to create your own custom admin or reset an admin password:
```bash
cd backend
# (Ensure virtual environment is activated)

# 1. Create a custom admin:
python scripts/create_admin.py --email myadmin@hospital.org --password "MySecretPass2026!" --name "Dr. Admin"

# 2. Reset password for an existing admin:
python scripts/create_admin.py --email admin@hospital.org --password "NewPassword2026!" --reset

# 3. List all existing administrators:
python scripts/create_admin.py --list
```

### How to Create a Doctor or Nurse Account
1. On the login screen, click **Register**.
2. Select your role: **Doctor** or **Nurse**.
3. In the **Staff Authorization Code** field, enter:  
   `RECOVERA-STAFF-2026`
4. Enter your name, email, department/specialization, and password.
5. Click **Register** — you can now log in with your new clinical account!

### How to Register a Patient Account
* Patients can self-register via the portal's Register page (select role **Patient**).
* Alternatively, Doctors and Nurses can admit/register new patients directly from the Doctor Dashboard.

---

## 🧪 7. Running Tests (Optional)

To verify the installation, activate the backend virtual environment and run pytest:

```bash
cd backend
python -m pytest ../tests/test_notification_extensions.py
python -m pytest ../tests/test_recovery_ai.py
```

---

## ❓ 8. Troubleshooting & FAQ

#### 1. PowerShell: "Running scripts is disabled on this system"
If you get an error when running `.\.venv\Scripts\Activate.ps1` in Windows PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
Then re-run `.\.venv\Scripts\Activate.ps1`.

#### 2. "Python is not recognized as an internal or external command"
Python is not added to your system `PATH`.
* Reinstall Python and make sure to check **"Add python.exe to PATH"**.
* Or add your Python installation path manually to your system Environment Variables.

#### 3. Port 8000 or 5173 is already in use
* Another process is running on port 8000 or 5173.
* On Windows, find and stop the process:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
  ```
  Or change the port in the startup command (e.g., `--port 8001`).

#### 4. Missing Gemini API Key
* The project does **not** fail if you do not have a Gemini API key.
* The built-in rule engine will automatically answer questions and manage recovery workflows safely.
* You can get a free key anytime from [Google AI Studio](https://aistudio.google.com/) and add it to `backend/.env`.

---

## 🛑 9. Stopping the Application

To shut down the servers at any time:
* Press **`Ctrl + C`** in both Terminal 1 and Terminal 2.
