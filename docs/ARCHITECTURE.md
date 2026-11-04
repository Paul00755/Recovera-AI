# Recovera AI — System Architecture (Version 3.0)

## 1. System Overview

Recovera AI is a hospital-supervised, post-discharge patient recovery management system. It provides clinicians (Doctors, Nurses, Admins) and Patients with structured clinical oversight, automated one-way SMS reminders, website-based medication intake tracking, conflict-free appointment scheduling, on-demand 15-minute consultations, daily symptom check-ins, and explainable recovery-risk prioritization.

```
+----------------------------------------------------------------------------------------------------+
|                                         Recovera AI Web UI                                         |
|  (React 19, Vite, Tailwind/CSS, Lucide, WebRTC, Web Speech Audio, Dark/Light responsive portals)   |
|                                                                                                    |
|    +------------------+  +------------------+  +------------------+  +------------------------+    |
|    | Patient Portal   |  | Doctor Workspace |  | Nurse Workspace  |  | Hospital Admin Portal  |    |
|    | - Intake tracker |  | - OCR approval   |  | - Reg & calendar |  | - User & role mgmt     |    |
|    | - Daily check-in |  | - Plan publisher |  | - 60m Appt book  |  | - Reset tickets        |    |
|    | - Grounded bot   |  | - Availability   |  | - Start Consult  |  | - SMS Gateway health   |    |
|    | - Audio & PDF    |  | - Risk overrides |  | - Patient search |  | - System audit logs    |    |
|    +------------------+  +------------------+  +------------------+  +------------------------+    |
+-------------------------------------------------+--------------------------------------------------+
                                                  | REST API & WebSockets (FastAPI / ASGI)
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                      Recovera AI Backend                                           |
|  (FastAPI, SQLAlchemy 2.0, APScheduler, WebRTC Signaling, ReportLab PDF, PaddleOCR Engine)         |
|                                                                                                    |
|  +--------------------+  +--------------------+  +--------------------+  +-----------------------+ |
|  | Auth & RBAC Core   |  | Clinical Documents |  | Medication Engine  |  | SMS Gateway Service   | |
|  | - JWT (HS256)      |  | - Upload validation|  | - Independent sched|  | - Pluggable drivers   | |
|  | - Bcrypt <72 bytes |  | - PaddleOCR clean  |  | - Dated occurrence |  | - OnePlus 10T HTTP    | |
|  | - Patient isolation|  | - Versioned review |  | - Taken/Not taken  |  | - Mock test driver    | |
|  | - Support tickets  |  | - Plan generation  |  | - Idempotency lock |  | - Bounded retries (3) | |
|  +--------------------+  +--------------------+  +--------------------+  +-----------------------+ |
|                                                                                                    |
|  +--------------------+  +--------------------+  +--------------------+  +-----------------------+ |
|  | Appointments Engine|  | Consultations & RTC|  | Risk Prioritization|  | Recovery Assistant    | |
|  | - Doctor status    |  | - 15m server timer |  | - Explainable rules|  | - Approved text RAG   | |
|  | - 60-min intervals |  | - WebRTC signaling |  | - Green/Yellow/Red |  | - Explicit uncertainty| |
|  | - Overlap rejection|  | - Room auth tokens |  | - Insufficient data|  | - Refuses diagnosis  | |
|  | - Transaction lock |  | - Real-time chat   |  | - Clinician override|  | - Speech TTS (EN/TA/HI| |
|  +--------------------+  +--------------------+  +--------------------+  +-----------------------+ |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                    +-----------------------------+-----------------------------+
                    |                                                           |
                    v                                                           v
+---------------------------------------+                   +---------------------------------------+
|          Relational Database          |                   |     OnePlus 10T 5G (Android 15)       |
| (SQLite dev / PostgreSQL prod via     |                   |       Primary Physical SIM Bridge     |
|  SQLAlchemy 2.0 ORM & Migrations)     |                   |  - Secured HTTP Gateway Companion     |
+---------------------------------------+                   |  - Token-authenticated REST API       |
                                                            |  - One-way Outbound SMS Dispatches    |
                                                            +---------------------------------------+
```

## 2. Core Architectural Principles

1. **One-Way SMS Notification Model**:
   - SMS is purely an outbound reminder and alerting channel via the OnePlus 10T 5G.
   - All patient responses, medication intake logging, and symptom reporting occur directly on the authenticated web application.
   - No SMS reply parsing, no Twilio WhatsApp sandbox, and no third-party messaging dependencies.

2. **Strict Clinical Safety Gate**:
   - Raw OCR extraction output is treated strictly as an unverified draft.
   - Clinician review and explicit Doctor approval are mandatory before any prescription or recovery plan becomes patient-visible.
   - When source documents change, dependent published recovery plans are immediately marked `NEEDS_REVIEW` and protected from patient visibility until re-approved.

3. **Independent Medication Schedules & Occurrences**:
   - Each medication maintains an independent start date, end date, frequency, and dose session definitions.
   - Individual dated `MedicationOccurrence` records are created for each scheduled dose (e.g. 3 days × 2 sessions = 6 occurrences).
   - Edits or discontinuations cancel only future pending occurrences; past adherence records are permanently preserved.

4. **Deterministic & Explainable Risk Prioritization**:
   - Risk categorization (`Green`, `Yellow`, `Red`) is calculated using explicit, clinician-reviewed rules evaluating actual adherence rates, missed doses, and symptom trends.
   - The engine explicitly outputs the triggering factors and data window.
   - If longitudinal data is insufficient (< 2 days), the status is reported explicitly as `Insufficient Data` without fabricating a risk score.
   - Clinicians may record overrides with compulsory clinical rationale in the audit trail.

5. **Server-Enforced Teleconsultation Lifecycle**:
   - Immediate consultations (audio, video, chat) are restricted to a maximum duration of 15 minutes.
   - The 15-minute countdown and termination are enforced server-side by the WebSocket signaling coordinator, preventing client-side bypass.
