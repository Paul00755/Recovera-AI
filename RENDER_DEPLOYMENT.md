# Recovera AI: Render Free Tier Deployment Guide

This guide provides step-by-step instructions to deploy Recovera AI (FastAPI Backend + React Frontend + Deep Learning Vision Models) on Render's 100% Free Tier.

---

## Architecture Overview on Render Free Tier

| Component | Render Service Type | Specs / Plan | Notes |
| :--- | :--- | :--- | :--- |
| **Backend API** | Web Service | Free Tier (512 MB RAM, Shared CPU) | Runs `uvicorn` with `onnxruntime` CPU (<45MB RAM footprint) |
| **Deep Learning Models** | Embedded in Repo | Lightweight ONNX | DenseNet-121 (99.2% Acc) & ResNet-50 (82.5% Acc, 95.0% Top-3) |
| **Frontend Portal** | Static Site | Free Tier (Unlimited bandwidth) | React + Vite + Tailwind SPA with instant global CDN |
| **Database** | Neon PostgreSQL | Free Tier (Serverless Postgres) | Automatic migrations on startup via SQLAlchemy |

> [!NOTE]
> **Why ONNX Runtime on Render?**
> Render free web services have a strict **512 MB RAM** limit. Loading full PyTorch CUDA or PaddlePaddle consumes >450 MB RAM, leading to out-of-memory `SIGKILL (OOM)` termination. 
> Recovera AI uses **ONNX Runtime (CPU)** with graph optimizations. Forward passes complete in **<60 ms** using only **~35 MB RAM**, guaranteeing stability on the free tier.

---

## Step 1: Create Free Neon PostgreSQL Database

1. Go to [Neon Console](https://console.neon.tech) and sign up (free).
2. Create a new project named `recovera-db`.
3. In the Dashboard, copy the connection string. It will look like:
   ```text
   postgresql://recovera_owner:abc123xyz@ep-summer-pond-123456.us-east-2.aws.neon.tech/recovera?sslmode=require
   ```
4. Recovera AI will automatically create all tables, indexes, and initial admin accounts on first startup.

---

## Step 2: Deploy Backend Web Service on Render

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repository: `Recovera AI`.
4. Configure the service settings:
   - **Name**: `recovera-ai-backend`
   - **Region**: Choose closest (e.g. `Oregon (US West)` or `Frankfurt (EU Central)`)
   - **Branch**: `main`
   - **Root Directory**: *(leave blank)*
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command**:
     ```bash
     cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type**: **Free** ($0/month)

5. Under **Environment Variables**, add the following keys:
   | Key | Value | Description |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.12.0` | Required Python runtime |
   | `ENVIRONMENT` | `production` | Enables production security |
   | `DATABASE_URL` | `postgresql://...` | Paste your Neon PostgreSQL connection string from Step 1 |
   | `SECRET_KEY` | *(Click "Generate")* | 32+ character JWT encryption key |
   | `CORS_ORIGINS` | `https://recovera-ai-frontend.onrender.com,http://localhost:5173` | Allowed frontend domains |
   | `UPLOAD_DIR` | `/tmp/storage` | Ephemeral or persistent storage path |
   | `ONEPLUS_GATEWAY_ENABLED` | `false` | Set to `false` for cloud deployment |

6. Click **Create Web Service**. Render will install dependencies and start the API.
7. Note down your backend URL (e.g., `https://recovera-ai-backend.onrender.com`).

---

## Step 3: Deploy Frontend Static Site on Render

1. In Render Dashboard, click **New +** -> **Static Site**.
2. Select your repository: `Recovera AI`.
3. Configure the frontend settings:
   - **Name**: `recovera-ai-frontend`
   - **Branch**: `main`
   - **Root Directory**: *(leave blank)*
   - **Build Command**:
     ```bash
     cd frontend && npm install && npm run build
     ```
   - **Publish Directory**:
     ```text
     frontend/dist
     ```
4. Under **Environment Variables**:
   | Key | Value |
   | :--- | :--- |
   | `VITE_API_URL` | `https://recovera-ai-backend.onrender.com` *(your backend URL from Step 2)* |
5. Under **Redirects / Rewrites**:
   - Add Rewrite Rule:
     - **Source**: `/*`
     - **Destination**: `/index.html`
     - **Action**: `Rewrite`
   *(This ensures client-side routing works for all URLs)*.
6. Click **Create Static Site**.

---

## Step 4: Verify Trained Deep Learning Vision Models

Once deployed, visit your frontend URL (or API `/docs`):
1. Navigate to **Medical Image Analysis** in the portal.
2. **Bone Fracture (X-Ray)**:
   - Upload any radiograph: The system runs DenseNet-121 (**99.21% test accuracy**), detects fractures, evaluates cortical margin displacement, and overlays the Grad-CAM activation map.
3. **Surgical Wound & Skin Analysis**:
   - Upload a clinical wound photograph: The system runs the 10-class model (**82.53% accuracy, 95.04% Top-3**), reports top differential diagnoses, calculates erythema and dehiscence risk scores, and generates the tissue segmentation mask.
4. **Initial Credentials**:
   - Initial Admin: `admin@hospital.org` / `AdminSecure2026!`
   *(You can create doctor and nurse accounts directly in the Admin portal)*.

---

## Optional: Deploy via Blueprint (`render.yaml`)

If you prefer 1-click deployment, Render can read the included `render.yaml`:
1. In Render Dashboard, click **New +** -> **Blueprint**.
2. Select the `Recovera AI` repo.
3. Render will automatically parse `render.yaml` and prompt you for `DATABASE_URL`.
4. Click **Apply**. Both backend and frontend will be provisioned simultaneously.
