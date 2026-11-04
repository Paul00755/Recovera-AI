# Recovera AI — Acceptance Test Matrix (Version 3.0)

## 1. Traceable Test Matrix

| ID | Test Scenario | Specification Baseline Reference | Target Component | Automated Test File | Pass Criteria |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T01** | Valid/invalid login & password byte limit (<72B) | Section 4, 18 | `auth_service` | `tests/test_auth_rbac.py` | Valid credentials return JWT; bcrypt rejects >72 UTF-8 bytes cleanly without 500 error. |
| **T02** | Reset ticket workflow (no unauthenticated reset) | Section 4 | `auth_service` | `tests/test_auth_rbac.py` | Ticket created with status `open`; Admin can resolve; no open password reset endpoint. |
| **T03** | Patient role isolation (patient cannot access staff APIs or other patients) | Section 3, 18 | `rbac_guard` | `tests/test_auth_rbac.py` | Returns 403 Forbidden on staff endpoints; patient isolation enforced on all patient-specific routes. |
| **T04** | Nurse role boundaries (nurse cannot perform admin actions or publish plans) | Section 3 | `rbac_guard` | `tests/test_auth_rbac.py` | Nurse forbidden from user management or setting plan status to `PUBLISHED`. |
| **T05** | Document upload, MIME check & noise-cleaned PaddleOCR extraction | Section 6 | `ocr_service` | `tests/test_documents_ocr.py` | Supported PDF/images accepted; headers, page numbers & noise filtered; structured draft created. |
| **T06** | Clinician review, draft edit, approval/rejection lifecycle | Section 6 | `documents_api` | `tests/test_documents_ocr.py` | Doctor approval updates status; rejected drafts hidden from patient portal. |
| **T07** | Recovery plan auto-generation & invalidation on source change | Section 7 | `recovery_service`| `tests/test_recovery_plan.py` | Changing approved source flags active plan `NEEDS_REVIEW`; hides patient view until reviewed. |
| **T08** | Medication occurrence generation (3d×2 sessions = 6, 5d×3 sessions = 15) | Section 8 | `medication_service`| `tests/test_medication_engine.py` | Exact occurrence counts generated with distinct date/session labels; no schedule cross-contamination. |
| **T09** | Medication edit & discontinuation with historical preservation | Section 8 | `medication_service`| `tests/test_medication_engine.py` | Discontinuation cancels future pending occurrences; prior taken/not-taken records are preserved. |
| **T10** | Website intake tracker (Taken / Not Taken + optional reason) | Section 8 | `medications_api` | `tests/test_patient_tracker.py` | Persists intake state with timestamp and optional reason; survives page reload. |
| **T11** | Registration welcome SMS dispatch | Section 9 | `sms_service` | `tests/test_sms_gateway.py` | Triggers welcome SMS only when patient has phone and opted-in. |
| **T12** | Medication dose reminder (30m prior to dose) | Section 9 | `scheduler_service`| `tests/test_sms_gateway.py` | Scheduled job queues at dose time - 30 minutes; uses approved prescription data. |
| **T13** | One-hour tracker follow-up SMS (1h after dose reminder) | Section 9 | `scheduler_service`| `tests/test_sms_gateway.py` | Follow-up job dispatched 60 minutes after dose reminder is accepted. |
| **T14** | SMS Gateway offline handling, bounded retry (max 3), idempotency | Section 9 | `sms_gateway` | `tests/test_sms_gateway.py` | OnePlus disconnect leaves jobs in QUEUED; 3 retries max before FAILED; duplicate prevention via key. |
| **T15** | Doctor Online/Away controls & published availability slots | Section 10 | `doctor_service` | `tests/test_appointments.py` | Doctor can toggle status and publish 60-minute time slots. |
| **T16** | 60-Minute conflict-free appointment booking with overlap rejection | Section 10 | `appointments_api`| `tests/test_appointments.py` | Concurrent or overlapping booking attempts for the same doctor are rejected with 409 Conflict. |
| **T17** | Immediate consultation orchestration with short-lived tokens | Section 11 | `consultations_api`| `tests/test_consultations.py` | Room created with 15-minute expiry; SMS join links sent to doctor and patient. |
| **T18** | Real WebRTC signaling & server-enforced 15-minute termination | Section 11 | `webrtc_signaling` | `tests/test_consultations.py` | WebSockets exchange SDP/ICE; server terminates connection and revokes access at 15 minutes. |
| **T19** | Daily symptom check-in submission & longitudinal tracking | Section 13 | `checkins_api` | `tests/test_checkins.py` | Captures pain level, location, sleep, activity, fever, cough; persists without synthetic fillers. |
| **T20** | Explainable Green/Yellow/Red risk calculation & Insufficient Data state | Section 14 | `risk_service` | `tests/test_risk_engine.py` | Explicit rules and reasons output; <2 days data yields `Insufficient Data`; clinician override logged. |
| **T21** | Patient PDF Report generation with truthful empty states | Section 14 | `report_service` | `tests/test_pdf_report.py` | Generates ReportLab PDF matching stored records; prints truthful messages for empty history. |
