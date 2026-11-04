# Recovera AI — Running Instructions (Your PC)

This guide contains the exact commands, comments, and configuration details to run **Recovera AI** on your machine (`D:\Recovera AI`). All virtual environments, dependencies, database files, and configurations are already pre-configured here.

---

## ⚡ Method 1: One-Click Quick Launch (Recommended)

You can launch both the **FastAPI Backend** and the **React/Vite Frontend** simultaneously in separate dedicated terminal windows:

1. Open File Explorer and navigate to `D:\Recovera AI`.
2. Double-click **`run.bat`**.

> **What `run.bat` does under the hood:**
> * Checks if Docker / PostgreSQL is running; if not, the system automatically uses your local SQLite database (`recovera.db`).
> * Launches the FastAPI Backend API on `http://127.0.0.1:8000` in a separate command window.
> * Launches the React/Vite Frontend Portal on `http://localhost:5173` in a second command window.
> * Leaves both servers running with live code reloading enabled.

---

## 💻 Method 2: Manual Terminal-by-Terminal Launch

If you prefer to start the servers manually or inspect logs in separate PowerShell or CMD terminals, follow these steps:

### Terminal 1: Backend API (FastAPI)

Open **PowerShell** or **Command Prompt** and run:

```powershell
# Step 1: Navigate to the backend directory
cd "d:\Recovera AI\backend"

# Step 2: Activate the Python virtual environment
.\.venv\Scripts\Activate.ps1
# (If using Command Prompt CMD instead of PowerShell, run: .venv\Scripts\activate.bat)

# Step 3: Start the Uvicorn server with hot-reload enabled
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

> **Direct One-Liner (No activation required):**
> ```powershell
> "d:\Recovera AI\backend\.venv\Scripts\python.exe" -m uvicorn app.main:app --app-dir "d:\Recovera AI\backend" --reload --host 127.0.0.1 --port 8000
> ```

* **Backend URL**: `http://127.0.0.1:8000`
* **Swagger API Docs**: `http://127.0.0.1:8000/docs`
* **Health Check**: `http://127.0.0.1:8000/api/health`

---

### Terminal 2: Frontend UI (React + Vite)

Open a **second terminal window** and run:

```powershell
# Step 1: Navigate to the frontend directory
cd "d:\Recovera AI\frontend"

# Step 2: Start the Vite development server
npm run dev
```

* **Frontend Web App URL**: `http://localhost:5173`

---

## ⚙️ Configuration & Environment Settings (`backend\.env`)

Your backend environment configuration is located at `D:\Recovera AI\backend\.env`. Here are the key variables:

```ini
# Database: Automatically probes PostgreSQL (5432) and seamlessly falls back to recovera.db
DATABASE_URL=postgresql+psycopg://recovera:change-me@localhost:5432/recovera

# JWT Security
JWT_SECRET=recovera-super-secret-jwt-key-2026-production
JWT_EXPIRE_MINUTES=120

# Staff Authorization Code for Doctor/Nurse Registration
STAFF_REGISTRATION_CODE=RECOVERA-STAFF-2026

# Google Gemini API for Recovery Assistant AI & Multilingual Support
# Add your Gemini API key here (get free key from https://aistudio.google.com/)
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash

#  SMS Gateway (Companion App)
ONEPLUS_GATEWAY_URL=http://192.0.0.2:8080/send-sms
ONEPLUS_GATEWAY_BACKUP_URL=http://100.88.34.95:8080
ONEPLUS_GATEWAY_TOKEN=oneplus-secret-companion-token-2026
ONEPLUS_GATEWAY_ENABLED=true
```

> [!NOTE]
> **SMS Gateway Handling:**
> If your OnePlus phone is disconnected or off the Wi-Fi network, the system will **not crash**. It logs a notice and marks the SMS jobs as queued/simulated, allowing consultations, logins, and test suites to run without interruption.

> [!TIP]
> **Gemini AI Key:**
> If `GEMINI_API_KEY` is left blank, the Recovery Assistant AI automatically uses its built-in rule-based fallback model. Adding a free Gemini key unlocks full generative analysis, multilingual translation (Tamil/Hindi), and deep clinical reasoning.

---

## 🔑 Default Accounts & Access Credentials

Once the servers are running, navigate to `http://localhost:5173`:

| Role | Email | Password | Notes |
| :--- | :--- | :--- | :--- |
| **System Admin** | `admin@hospital.org` | `AdminSecure2026!` | Primary pre-seeded administrator |
| **Admin** | `paul123@gmail.com` | *(Your configured password)* | Secondary administrator |
| **Register Staff** | *(New Doctor / Nurse)* | *(Your choice)* | Click **Register** on portal, select Doctor/Nurse, enter Staff Code: `RECOVERA-STAFF-2026` |
| **Patient** | *(Self-register or admit)* | *(Your choice)* | Register via portal or admit through Doctor Dashboard |

---

## 🗄️ Database Management & Maintenance

* **Current Database**: `D:\Recovera AI\recovera.db` (SQLite canonical database with automatic schema migrations on startup).
* **Resetting / Clearing Test Users & Patients**:
  If you ever want to clear test data and start fresh:
  ```powershell
  & "d:\Recovera AI\backend\.venv\Scripts\python.exe" "d:\Recovera AI\backend\scripts\clear_users_data.py"
  ```
* **Creating / Managing Admin Accounts**:
  ```powershell
  # 1. List existing admins:
  & "d:\Recovera AI\backend\.venv\Scripts\python.exe" "d:\Recovera AI\backend\scripts\create_admin.py" --list

  # 2. Create a default admin (admin@hospital.org / AdminSecure2026!):
  & "d:\Recovera AI\backend\.venv\Scripts\python.exe" "d:\Recovera AI\backend\scripts\create_admin.py"

  # 3. Create a custom admin with your own email and password:
  & "d:\Recovera AI\backend\.venv\Scripts\python.exe" "d:\Recovera AI\backend\scripts\create_admin.py" --email myadmin@example.com --password "MySecurePassword2026!" --name "Dr. Admin"

  # 4. Reset an existing admin's password:
  & "d:\Recovera AI\backend\.venv\Scripts\python.exe" "d:\Recovera AI\backend\scripts\create_admin.py" --email admin@hospital.org --password "NewPassword2026!" --reset
  ```
* **Optional PostgreSQL Container**:
  If you want to run PostgreSQL instead of SQLite:
  ```powershell
  docker compose up -d
  ```

---

## 🧪 Running Automated Tests

To verify that all notifications, AI endpoints, and clinical features are functioning properly:

```powershell
# Test SMS & In-App Notifications (Consultation calls + 7 clinical suggestions)
& "d:\Recovera AI\backend\.venv\Scripts\python.exe" -m pytest tests/test_notification_extensions.py

# Test Recovery AI Assistant (Medication query, safety boundary, multilingual, fallback)
& "d:\Recovera AI\backend\.venv\Scripts\python.exe" -m pytest tests/test_recovery_ai.py

# Test Medication Schedules & SMS Gateway Reminders
& "d:\Recovera AI\backend\.venv\Scripts\python.exe" -m pytest tests/test_phase6_phase7_sms_reminders.py
```

---

## 🛑 Stopping the Application

To shut down the servers:
* If running via `run.bat`: Close both the Backend and Frontend terminal windows, or press `Ctrl + C` in each window.
* If running manually: Press `Ctrl + C` in the respective PowerShell / CMD windows.
