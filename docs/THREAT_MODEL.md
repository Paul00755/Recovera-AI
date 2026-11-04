# Recovera AI — Threat Model & Security Architecture (Version 3.0)

## 1. System Assets & Sensitivity

1. **Protected Health Information (PHI)**: Patient demographics, medical history, prescriptions, discharge summaries, recovery plans, daily symptom reports, doctor notes.
2. **Access Credentials & Session Tokens**: Passwords, JWT bearer tokens, short-lived consultation room join tokens.
3. **SMS Gateway Bridge**: Local network HTTP API credentials connecting to the OnePlus 10T 5G.
4. **Clinical Instructions & Treatment Integrity**: Prescriptions and instructions delivered to recovering patients.

## 2. Threat Analysis & Mitigations

### 2.1 Role-Based Access Control (RBAC) & Privilege Escalation
- **Threat**: Malicious user attempts to perform unauthorized actions (e.g. nurse approves doctor prescriptions; patient views other patients' records).
- **Mitigation**:
  - Backend enforcement on every endpoint using dependency guards (`require_role(Role.DOCTOR)`, `require_role(Role.ADMIN)`).
  - Explicit patient isolation checks: A user with `role="patient"` is verified against `current_patient.id == target_patient.id`. Attempts to access mismatched records return `403 Forbidden`.
  - UI role filtering is purely presentation-layer; backend never relies on frontend hiding.

### 2.2 Account Takeover & Password Attacks
- **Threat**: Brute-force guessing, dictionary attacks, or password truncation vulnerabilities.
- **Mitigation**:
  - Passwords hashed with bcrypt (`passlib[bcrypt]`, pinned `bcrypt==4.0.1`).
  - Strict byte-length validation: Passwords exceeding 72 UTF-8 bytes are rejected during validation before reaching the bcrypt hashing function.
  - No public self-service password reset. "Forgot Password" creates a `PasswordResetTicket` that requires authenticated hospital Admin approval.
  - Staff registration strictly gated by secret hospital code (`STAFF_REGISTRATION_CODE`).

### 2.3 Clinical Information Poisoning & Raw OCR Ingestion
- **Threat**: Unverified or hallucinated OCR data shown to patients as clinical advice.
- **Mitigation**:
  - Raw OCR extraction output is strictly classified as `DRAFT`.
  - Explicit Doctor approval is required to publish recovery plans or activate medication schedules.
  - Unapproved or rejected records are filtered out of patient-facing queries at the database query layer.
  - Changes to underlying source documents automatically invalidate published plans to `NEEDS_REVIEW` and hide them until re-approved.

### 2.4 SMS Gateway & Telephony Abuse
- **Threat**: Unauthorized entities sending arbitrary SMS messages through the hospital phone or spamming patients.
- **Mitigation**:
  - The phone HTTP companion accepts requests exclusively from trusted internal backend IPs with bearer token validation (`X-Gateway-Token`).
  - Strict daily transmission volume cap (default 500) and rate limit (1 SMS per 3 seconds).
  - One-way notification policy: SMS inbound replies are completely ignored and cannot trigger state mutations.
  - Notification opt-in consent recorded and checked prior to queuing any message.

### 2.5 Teleconsultation Room Hijacking & Token Replay
- **Threat**: Unauthorized participants joining private doctor-patient video/audio/chat consultations.
- **Mitigation**:
  - Consultation rooms use unguessable UUIDs.
  - Short-lived room JWT tokens (15-minute expiration) are signed by backend and tied to specific `user_id` and `room_id`.
  - WebSockets authenticate tokens upon handshake.
  - Signaling server tracks participants and enforces server-side disconnection at exactly 15 minutes.
  - Revoked/completed rooms cannot be rejoined.
