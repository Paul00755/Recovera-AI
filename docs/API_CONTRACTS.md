# Recovera AI — REST & WebSocket API Contracts (Version 3.0)

## 1. Authentication & Security Endpoints (`/api/auth`)

### `POST /api/auth/login`
- **Description**: Authenticates Doctor, Nurse, Admin, or Patient.
- **Request**:
  ```json
  { "email": "doctor@hospital.org", "password": "SecurePassword123!" }
  ```
- **Response** (200 OK):
  ```json
  { "token": "<JWT_HS256>", "user": { "id": "u-1", "email": "doctor@hospital.org", "name": "Dr. Sarah Rao", "role": "doctor" } }
  ```

### `POST /api/auth/register-staff`
- **Description**: Controlled registration for Doctors and Nurses using hospital code.
- **Request**:
  ```json
  { "email": "nurse.priya@hospital.org", "password": "SecurePassword123!", "name": "Priya Sharma", "role": "nurse", "staff_code": "RECOVERA-STAFF-2026" }
  ```
- **Response** (201 Created):
  ```json
  { "id": "u-2", "email": "nurse.priya@hospital.org", "name": "Priya Sharma", "role": "nurse" }
  ```

### `POST /api/auth/forgot-password`
- **Description**: Creates a password-reset support ticket for hospital Admin review (no unrestricted reset).
- **Request**:
  ```json
  { "email": "user@hospital.org", "reason": "Forgot credential" }
  ```
- **Response** (200 OK):
  ```json
  { "ticket_id": "ticket-101", "status": "open", "message": "Password reset ticket submitted for Admin review." }
  ```

---

## 2. Patients & Medical History Endpoints (`/api/patients`)

### `GET /api/patients`
- **Description**: Searchable patient directory for authorized clinicians.
- **Query Params**: `q` (search term), `page`, `limit`
- **Response** (200 OK): List of patient summaries.

### `POST /api/patients`
- **Description**: Staff registers a new patient with normalized phone and SMS notification preferences.
- **Request**:
  ```json
  {
    "name": "Ravi Kumar",
    "dob": "1980-05-15",
    "phone": "+919876543210",
    "email": "ravi.kumar@example.com",
    "preferred_language": "en",
    "sms_opt_in": true,
    "has_medical_history": false,
    "medical_history": null
  }
  ```
- **Response** (201 Created):
  ```json
  { "id": "p-101", "patient_code": "P261012A1B2", "name": "Ravi Kumar", "sms_opt_in": true }
  ```

### `GET /api/patients/{pid}/medical-history`
- **Description**: Returns structured past conditions, medications, treatments, and hospitalizations, or truthful "No previous medical history recorded".

---

## 3. Clinical Documents & PaddleOCR (`/api/documents`)

### `POST /api/patients/{pid}/documents`
- **Description**: Upload prescription, discharge summary, or recovery instructions (PDF/images).
- **Form Data**: `file` (multipart/form-data), `doc_type` ("prescription", "discharge_summary", "recovery_instructions", "follow_up_schedule")
- **Response** (201 Created): Includes document ID, status `DRAFT`, and trigger for OCR extraction.

### `GET /api/documents/{doc_id}/ocr`
- **Description**: Returns cleaned PaddleOCR extraction with confidence levels and draft fields.

### `PATCH /api/documents/{doc_id}/review`
- **Description**: Clinician corrects draft fields and sets status: `APPROVED` or `REJECTED`.

---

## 4. Medication Engine & Adherence Tracker (`/api/medications`)

### `GET /api/patients/{pid}/medications`
- **Description**: Active and historical prescribed medications.

### `GET /api/patients/{pid}/medication-occurrences`
- **Description**: Scheduled dated occurrences for today and past windows.
- **Response** (200 OK):
  ```json
  [
    {
      "id": "occ-101",
      "medication_id": "med-1",
      "medicine_name": "Amoxicillin",
      "dose": "500mg",
      "scheduled_at": "2026-10-12T08:00:00Z",
      "state": "pending",
      "food_relation": "after_food"
    }
  ]
  ```

### `POST /api/medication-occurrences/{occ_id}/adherence`
- **Description**: Patient records intake status on the website.
- **Request**:
  ```json
  { "state": "not_taken", "reason": "felt unwell" }
  ```
- **Response** (200 OK):
  ```json
  { "occurrence_id": "occ-101", "state": "not_taken", "recorded_at": "2026-10-12T08:45:00Z" }
  ```

---

## 5. Doctor Availability & Appointments (`/api/appointments`)

### `GET /api/doctors/{did}/availability`
- **Description**: Published 60-minute time slots and current status (`Online` / `Away`).

### `POST /api/appointments`
- **Description**: Nurse schedules a 60-minute conflict-free appointment.
- **Request**:
  ```json
  { "patient_id": "p-101", "doctor_id": "doc-1", "start_time": "2026-10-13T10:00:00Z" }
  ```
- **Response** (201 Created or 409 Conflict if slot is already occupied).

---

## 6. Consultations & WebRTC Signaling (`/api/consultations` & `/ws/consultation/{room_id}`)

### `POST /api/consultations/immediate`
- **Description**: Nurse starts an immediate 15-minute consultation with an Online doctor.
- **Request**:
  ```json
  { "patient_id": "p-101", "doctor_id": "doc-1", "mode": "video" }
  ```
- **Response** (201 Created):
  ```json
  {
    "room_id": "rm-uuid-99",
    "mode": "video",
    "doctor_join_url": "http://.../consultation/rm-uuid-99?token=<doc_token>",
    "patient_join_url": "http://.../consultation/rm-uuid-99?token=<patient_token>",
    "expires_at": "2026-10-12T20:00:00Z"
  }
  ```

### `WebSocket /ws/consultation/{room_id}?token=<jwt>`
- **Description**: Real-time signaling protocol:
  - Messages: `join`, `offer`, `answer`, `ice_candidate`, `chat_message`, `session_timer_tick`, `session_expired`.
  - Server automatically closes connection and marks session terminated at exactly 15 minutes.

---

## 7. Daily Check-ins, Risk Engine & PDF (`/api/check-ins`, `/api/risk`, `/api/reports`)

### `POST /api/patients/{pid}/check-ins`
- **Request**:
  ```json
  { "pain_level": 4, "pain_location": "lower abdomen", "sleep_hours": 7.0, "activity_level": "moderate", "fever": false, "cough": false, "notes": "Feeling slightly better" }
  ```

### `GET /api/patients/{pid}/risk-classification`
- **Response** (200 OK):
  ```json
  {
    "level": "green",
    "explainability_reasons": ["Medication adherence is 92% over last 7 days", "Reported pain severity is mild (<= 4)"],
    "insufficient_data": false,
    "evaluated_at": "2026-10-12T19:45:00Z"
  }
  ```

### `GET /api/patients/{pid}/report.pdf`
- **Description**: Streams patient-specific ReportLab PDF with truthful empty states and zero synthetic filler data.
