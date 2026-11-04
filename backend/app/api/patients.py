import uuid
import re
import datetime as dt
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, or_, func

from app.core.config import settings
from app.core.database import get_db
from app.core.constants import Role, AppointmentStatus, RiskLevel, DocStatus
from app.core.audit import record_audit
from app.models.patient import Patient, MedicalHistory, PatientDoctorAssignment, PatientDeletionTicket, PatientReactivationTicket
from app.models.user import User
from app.models.appointment import Appointment
from app.schemas.patient import (
    PatientCreateIn,
    PatientUpdateIn,
    MedicalHistoryIn,
    AssignDoctorRequest,
    PatientPreferencesIn,
    PatientConditionUpdateIn,
    PatientDeletionTicketCreateIn,
    PatientDeletionTicketReviewIn,
    CareStatusUpdateIn,
    ReactivationTicketCreateIn,
    ReactivationTicketReviewIn,
)
from app.api.deps import get_current_user, require_role, enforce_patient_isolation

router = APIRouter(prefix="/api/patients", tags=["Patient Workspace & Directory"])

def normalize_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    cleaned = re.sub(r"[\s\-\(\)]", "", phone.strip())
    return cleaned

def is_doctor_assigned_to_patient(doctor_id: str, patient_id: str, db: Session) -> bool:
    patient = db.get(Patient, patient_id)
    if not patient:
        return False
    if patient.assigned_doctor_id == doctor_id:
        return True
    active_assign = db.scalar(
        select(PatientDoctorAssignment).where(
            PatientDoctorAssignment.patient_id == patient_id,
            PatientDoctorAssignment.doctor_id == doctor_id,
            PatientDoctorAssignment.is_active == True
        )
    )
    return active_assign is not None

def patient_to_dict(p: Patient, db: Session) -> dict:
    user = p.user if p.user else db.get(User, p.user_id) if p.user_id else None
    assigned_doc = db.get(User, p.assigned_doctor_id) if p.assigned_doctor_id else None

    # Calculate age
    age = None
    if p.dob:
        today = dt.date.today()
        age = today.year - p.dob.year - ((today.month, today.day) < (p.dob.month, p.dob.day))

    # Assigned doctors list
    assignments = db.scalars(
        select(PatientDoctorAssignment).where(
            PatientDoctorAssignment.patient_id == p.id,
            PatientDoctorAssignment.is_active == True
        )
    ).all()

    assigned_doctors = []
    for a in assignments:
        doc = a.doctor or db.get(User, a.doctor_id)
        if doc:
            assigned_doctors.append({
                "assignment_id": a.id,
                "doctor_id": doc.id,
                "name": doc.name,
                "specialization": doc.specialization,
                "department": a.department or doc.specialization,
                "phone": doc.phone,
                "email": doc.email,
                "assigned_at": a.assigned_at.isoformat() if a.assigned_at else None,
                "notes": a.notes
            })

    if assigned_doc and not any(d["doctor_id"] == assigned_doc.id for d in assigned_doctors):
        assigned_doctors.insert(0, {
            "assignment_id": None,
            "doctor_id": assigned_doc.id,
            "name": assigned_doc.name,
            "specialization": assigned_doc.specialization,
            "department": assigned_doc.specialization,
            "phone": assigned_doc.phone,
            "email": assigned_doc.email,
            "assigned_at": p.created_at.isoformat() if p.created_at else None,
            "notes": "Primary Physician"
        })

    primary_doc_name = assigned_doctors[0]["name"] if assigned_doctors else (assigned_doc.name if assigned_doc else None)
    primary_doc_spec = assigned_doctors[0]["specialization"] if assigned_doctors else (assigned_doc.specialization if assigned_doc else None)
    doctor_names_summary = ", ".join([d["name"] for d in assigned_doctors]) if assigned_doctors else "Unassigned"

    return {
        "id": p.id,
        "patient_code": p.patient_code,
        "name": user.name if user else "Patient",
        "dob": p.dob.isoformat() if p.dob else None,
        "age": age,
        "gender": p.gender or "unspecified",
        "phone": p.phone or (user.email if user else None),
        "email": user.email if user else None,
        "preferred_language": p.preferred_language,
        "sms_language": getattr(p, "sms_language", None) or p.preferred_language or "en",
        "sms_opt_in": p.sms_opt_in,
        "sms_opt_in_at": p.sms_opt_in_at.isoformat() if p.sms_opt_in_at else None,
        "previous_history": p.previous_history,
        "active": p.active,
        "care_status": getattr(p, "care_status", "active") or "active",
        "active_condition": getattr(p, "active_condition", None),
        "condition_stage": getattr(p, "condition_stage", "active"),
        "condition_notes": getattr(p, "condition_notes", ""),
        "condition_updated_at": p.condition_updated_at.isoformat() if getattr(p, "condition_updated_at", None) else None,
        "assigned_doctor_id": p.assigned_doctor_id or (assigned_doctors[0]["doctor_id"] if assigned_doctors else None),
        "assigned_doctor_name": primary_doc_name,
        "assigned_doctor_specialization": primary_doc_spec,
        "assigned_doctors": assigned_doctors,
        "doctor_names_summary": doctor_names_summary,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }

@router.get("")
def list_patients(
    q: Optional[str] = Query(None, description="Search by name, patient_code, phone, or email"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # If patient role, only return self
    if current_user.role == Role.PATIENT.value:
        p = db.scalar(select(Patient).where(Patient.user_id == current_user.id, Patient.active == True))
        return [patient_to_dict(p, db)] if p else []

    # Doctors only see their assigned patients (either direct or via patient_doctor_assignments)
    if current_user.role == Role.DOCTOR.value:
        subq = select(PatientDoctorAssignment.patient_id).where(
            PatientDoctorAssignment.doctor_id == current_user.id,
            PatientDoctorAssignment.is_active == True
        )
        query = (
            select(Patient)
            .where(
                Patient.active == True,
                or_(
                    Patient.assigned_doctor_id == current_user.id,
                    Patient.id.in_(subq)
                )
            )
            .order_by(Patient.created_at.desc())
        )
    else:
        # Admin / Nurse see all active patients
        query = select(Patient).where(Patient.active == True).order_by(Patient.created_at.desc())

    patients = db.scalars(query).all()

    results = []
    search_term = q.strip().lower() if q else None

    for p in patients:
        p_dict = patient_to_dict(p, db)
        if search_term:
            name_match = search_term in (p_dict["name"] or "").lower()
            code_match = search_term in (p_dict["patient_code"] or "").lower()
            phone_match = search_term in (p_dict["phone"] or "").lower()
            email_match = search_term in (p_dict["email"] or "").lower()
            if not (name_match or code_match or phone_match or email_match):
                continue
        results.append(p_dict)

    return results

@router.post("", status_code=status.HTTP_201_CREATED)
def create_patient(
    req: PatientCreateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    normalized_phone_val = normalize_phone(req.phone)
    if req.sms_opt_in and not normalized_phone_val:
        raise HTTPException(
            status_code=422,
            detail="A valid phone number is required when SMS notifications are enabled."
        )

    # 1. Create User account for patient with configured or default temporary password
    patient_email = req.email.strip().lower() if req.email else f"patient_{uuid.uuid4().hex[:8]}@hospital.local"
    existing_user = db.scalar(select(User).where(User.email == patient_email))
    if existing_user:
        raise HTTPException(status_code=409, detail=f"User with email '{patient_email}' already exists.")

    from app.core.security import get_password_hash, validate_password_length
    initial_raw_pass = req.password.strip() if req.password and req.password.strip() else "PatientTempPass2026!"
    if req.password and req.password.strip():
        validate_password_length(initial_raw_pass)

    user = User(
        email=patient_email,
        password_hash=get_password_hash(initial_raw_pass),
        name=req.name.strip(),
        role=Role.PATIENT.value,
        active=True,
        must_change_password=req.must_change_password,
        phone=normalized_phone_val
    )
    db.add(user)
    db.flush()

    # 2. Compute date of birth if age is provided
    patient_dob = req.dob
    if not patient_dob and req.age is not None and req.age > 0:
        today = dt.date.today()
        patient_dob = dt.date(today.year - req.age, today.month, min(today.day, 28))

    # 3. Generate patient_code e.g. P261012A1B2
    today_code = dt.datetime.utcnow().strftime("%y%m%d")
    unique_suffix = uuid.uuid4().hex[:4].upper()
    patient_code = f"P{today_code}{unique_suffix}"

    patient = Patient(
        user_id=user.id,
        patient_code=patient_code,
        dob=patient_dob,
        gender=req.gender or "unspecified",
        phone=normalized_phone_val,
        preferred_language=req.preferred_language or "en",
        sms_language=getattr(req, "sms_language", None) or req.preferred_language or "en",
        active=True,
        sms_opt_in=req.sms_opt_in,
        sms_opt_in_at=dt.datetime.utcnow() if req.sms_opt_in else None,
        sms_opt_in_recorded_by=current_user.id if req.sms_opt_in else None,
        previous_history=req.has_medical_history,
        created_by=current_user.id,
        created_at=dt.datetime.utcnow(),
        assigned_doctor_id=req.assigned_doctor_ids[0] if req.assigned_doctor_ids else None
    )
    db.add(patient)
    db.flush()

    # 4. Multi-doctor assignments
    if req.assigned_doctor_ids:
        from app.services.notification_service import create_in_app_notification
        for doc_id in req.assigned_doctor_ids:
            doc = db.get(User, doc_id)
            if doc and doc.role == Role.DOCTOR.value:
                assignment = PatientDoctorAssignment(
                    patient_id=patient.id,
                    doctor_id=doc.id,
                    assigned_by=current_user.id,
                    department=doc.specialization,
                    is_active=True,
                    assigned_at=dt.datetime.utcnow()
                )
                db.add(assignment)
                patient_display_name = req.name.strip() if req.name else patient.name
                create_in_app_notification(
                    db=db,
                    user_id=doc.id,
                    title="New Patient Assigned",
                    message=f"Patient {patient_display_name} has been assigned to you by {current_user.name}.",
                    category="assignment",
                    link=f"#/staff/patients/{patient.id}/overview"
                )

    # 3. Create structured MedicalHistory record
    med_hist = MedicalHistory(
        patient_id=patient.id,
        has_history=req.has_medical_history,
        conditions=req.medical_history.conditions if req.medical_history else [],
        past_medications=req.medical_history.past_medications if req.medical_history else [],
        past_treatments=req.medical_history.past_treatments if req.medical_history else [],
        past_hospitalizations=req.medical_history.past_hospitalizations if req.medical_history else [],
        notes=req.medical_history.notes if req.medical_history else "",
        updated_at=dt.datetime.utcnow()
    )
    db.add(med_hist)

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_registered",
        target_type="patient",
        target_id=patient.id,
        new_values={"name": req.name, "patient_code": patient_code, "sms_opt_in": req.sms_opt_in}
    )
    db.commit()

    # Phase 6: Automated Welcome SMS upon verified registration with notification opt-in
    if req.sms_opt_in and normalized_phone_val:
        from app.services.sms_gateway_service import queue_and_send_sms
        from app.core.constants import SMSCategory
        welcome_text = (
            f"Welcome to Recovera AI, {req.name.strip()}. Your hospital recovery oversight is active. "
            f"Access your recovery plan and medication schedule at: {settings.FRONTEND_URL.rstrip('/')}"
        )
        try:
            queue_and_send_sms(
                db=db,
                phone=normalized_phone_val,
                message_text=welcome_text,
                category=SMSCategory.REGISTRATION.value,
                patient_id=patient.id,
                idempotency_key=f"welcome_{patient.id}"
            )
        except Exception:
            pass

    return patient_to_dict(patient, db)

# ---------------------------------------------------------------------------
# GET /api/patients/nurse-dashboard – summary dashboard (nurse/admin only)
# MUST BE DEFINED BEFORE /{pid} so FastAPI does not interpret 'nurse-dashboard' as a pid!
# ---------------------------------------------------------------------------

@router.get("/nurse-dashboard")
def nurse_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.NURSE.value)),
):
    today = dt.date.today()
    today_start = dt.datetime.combine(today, dt.time.min)
    today_end = dt.datetime.combine(today, dt.time.max)
    now = dt.datetime.utcnow()

    # All active patients
    all_patients_orm = db.scalars(
        select(Patient).where(Patient.active == True).order_by(Patient.created_at.desc())
    ).all()
    all_patients = [patient_to_dict(p, db) for p in all_patients_orm]

    # Preload all doctors into a map to prevent N queries
    all_docs = {u.id: u for u in db.scalars(select(User).where(User.role == Role.DOCTOR.value)).all()}

    # Group by specialty (via resolved assigned doctor's specialization)
    by_specialty: dict = {}
    for p_dict in all_patients:
        spec = p_dict.get("assigned_doctor_specialization") or ("unassigned" if not p_dict.get("assigned_doctor_id") else "unspecified")
        by_specialty.setdefault(spec, []).append(p_dict)

    # Preload latest risk classification for all active patients
    from app.models.risk_history import RiskClassificationHistory
    from app.models.digital_twin import DigitalTwinDeviationAlert
    from app.services.risk_service import evaluate_patient_risk

    active_pids = [p.id for p in all_patients_orm]
    latest_risks = {}
    if active_pids:
        risk_rows = db.execute(
            select(
                RiskClassificationHistory.patient_id,
                RiskClassificationHistory.level,
                RiskClassificationHistory.explainability_reasons
            )
            .where(RiskClassificationHistory.patient_id.in_(active_pids))
            .order_by(RiskClassificationHistory.patient_id, RiskClassificationHistory.evaluated_at.desc())
        ).all()
        for pid, lvl, reasons in risk_rows:
            if pid not in latest_risks:
                latest_risks[pid] = (lvl, reasons or [])

    critical_alert_pids = set(
        db.scalars(
            select(DigitalTwinDeviationAlert.patient_id)
            .where(
                DigitalTwinDeviationAlert.patient_id.in_(active_pids),
                DigitalTwinDeviationAlert.severity == "critical",
                DigitalTwinDeviationAlert.status == "pending"
            )
        ).all()
    ) if active_pids else set()

    # Clinical Triage & Urgency Stratification
    needs_immediate_care = []
    acuity_dist = {
        "critical_red": 0,
        "moderate_yellow": 0,
        "ambulatory_stable": 0,
        "unassigned_triage": 0,
        "total_active_patients": len(all_patients),
    }

    for p_dict in all_patients:
        pid = p_dict["id"]
        # If no risk record exists, evaluate on the fly
        if pid not in latest_risks:
            try:
                eval_hist = evaluate_patient_risk(db, pid)
                latest_risks[pid] = (eval_hist.level, eval_hist.explainability_reasons or [])
            except Exception:
                latest_risks[pid] = (RiskLevel.GREEN.value, [])

        risk_lvl, reasons = latest_risks.get(pid, (RiskLevel.GREEN.value, []))
        p_dict["risk_level"] = risk_lvl
        p_dict["risk_reasons"] = reasons
        has_crit_alert = pid in critical_alert_pids
        p_dict["has_critical_alert"] = has_crit_alert

        is_unassigned = not p_dict.get("assigned_doctor_id")
        is_red = (str(risk_lvl).lower() == "red")
        is_yellow = (str(risk_lvl).lower() == "yellow")

        # Clinical urgency triage classification
        if is_red:
            p_dict["urgency_reason"] = "Critical Clinical Risk (Red Status)"
            needs_immediate_care.append(p_dict)
            acuity_dist["critical_red"] += 1
        elif has_crit_alert:
            p_dict["urgency_reason"] = "Critical Trajectory Deviation Alert"
            needs_immediate_care.append(p_dict)
            acuity_dist["critical_red"] += 1
        elif is_unassigned:
            p_dict["urgency_reason"] = "Unassigned Doctor (Needs Triage)"
            needs_immediate_care.append(p_dict)
            acuity_dist["unassigned_triage"] += 1
        elif is_yellow:
            acuity_dist["moderate_yellow"] += 1
        else:
            acuity_dist["ambulatory_stable"] += 1

    # Doctor Caseload Balancing Distribution
    doc_pt_counts = {doc_id: 0 for doc_id in all_docs.keys()}
    unassigned_count = 0
    for p_dict in all_patients:
        doc_id = p_dict.get("assigned_doctor_id")
        if doc_id and doc_id in doc_pt_counts:
            doc_pt_counts[doc_id] += 1
        elif not doc_id:
            unassigned_count += 1

    doctor_caseload = []
    for doc_id, doc in all_docs.items():
        doctor_caseload.append({
            "doctor_id": doc_id,
            "name": f"Dr. {doc.name}",
            "specialization": doc.specialization or "General Medicine",
            "patient_count": doc_pt_counts.get(doc_id, 0)
        })
    if unassigned_count > 0:
        doctor_caseload.append({
            "doctor_id": None,
            "name": "Unassigned Specialist",
            "specialization": "Triage",
            "patient_count": unassigned_count
        })

    # Today's appointments (active, non-cancelled)
    todays_appts_orm = db.scalars(
        select(Appointment)
        .where(
            Appointment.start_time >= today_start,
            Appointment.start_time <= today_end,
            Appointment.status != AppointmentStatus.CANCELLED.value,
        )
        .order_by(Appointment.start_time.asc())
    ).all()

    # Missed appointments
    missed_appts_orm = db.scalars(
        select(Appointment)
        .where(Appointment.status == AppointmentStatus.MISSED.value)
        .order_by(Appointment.start_time.desc())
    ).all()

    # Upcoming appointments (scheduled/rescheduled, starting after now)
    upcoming_appts_orm = db.scalars(
        select(Appointment)
        .where(
            Appointment.status.in_([AppointmentStatus.SCHEDULED.value, AppointmentStatus.RESCHEDULED.value]),
            Appointment.start_time > now,
        )
        .order_by(Appointment.start_time.asc())
    ).all()

    # Batch preload patients and users for all appointment lists
    appts_all = todays_appts_orm + missed_appts_orm + upcoming_appts_orm
    pt_ids = {a.patient_id for a in appts_all}
    doc_ids = {a.doctor_id for a in appts_all}
    pts_map = {p.id: p for p in db.scalars(select(Patient).where(Patient.id.in_(pt_ids))).all()} if pt_ids else {}
    user_ids = {p.user_id for p in pts_map.values() if p.user_id}.union(doc_ids)
    users_map = {u.id: u for u in db.scalars(select(User).where(User.id.in_(user_ids))).all()} if user_ids else {}

    def _appt_brief(a: Appointment) -> dict:
        patient = pts_map.get(a.patient_id)
        doctor = users_map.get(a.doctor_id)
        patient_user = users_map.get(patient.user_id) if patient and patient.user_id else None
        return {
            "id": a.id,
            "patient_id": a.patient_id,
            "patient_name": patient_user.name if patient_user else (patient.patient_code if patient else "Unknown"),
            "doctor_id": a.doctor_id,
            "doctor_name": doctor.name if doctor else "Unknown",
            "doctor_specialization": doctor.specialization if doctor else None,
            "start_time": a.start_time.isoformat(),
            "end_time": a.end_time.isoformat(),
            "status": a.status,
            "appointment_type": getattr(a, "appointment_type", "follow_up") or "follow_up",
        }

    todays_appointments = [_appt_brief(a) for a in todays_appts_orm]
    missed_appointments = [_appt_brief(a) for a in missed_appts_orm]
    upcoming_appointments = [_appt_brief(a) for a in upcoming_appts_orm]

    # Real document verification funnel counts
    from app.models.clinical_document import ClinicalDocument, OCRExtraction

    total_docs = db.scalar(select(func.count(ClinicalDocument.id))) or 0
    draft_docs = db.scalar(
        select(func.count(ClinicalDocument.id)).where(ClinicalDocument.status == DocStatus.DRAFT.value)
    ) or 0
    ocr_count = db.scalar(select(func.count(OCRExtraction.id))) or 0
    under_review_docs = db.scalar(
        select(func.count(ClinicalDocument.id)).where(
            ClinicalDocument.status.in_([DocStatus.UNDER_REVIEW.value, DocStatus.NEEDS_REVIEW.value])
        )
    ) or 0
    approved_docs = db.scalar(
        select(func.count(ClinicalDocument.id)).where(ClinicalDocument.status == DocStatus.APPROVED.value)
    ) or 0

    return {
        "all_patients": all_patients,
        "by_specialty": by_specialty,
        "needs_immediate_care": needs_immediate_care,
        "acuity_distribution": acuity_dist,
        "doctor_caseload": doctor_caseload,
        "todays_appointments": todays_appointments,
        "missed_appointments": missed_appointments,
        "upcoming_appointments": upcoming_appointments,
        "document_funnel": {
            "total": total_docs,
            "draft": draft_docs,
            "ocr_extracted": ocr_count,
            "under_review": under_review_docs,
            "approved": approved_docs,
        }
    }


def _perform_patient_deletion(patient: Patient, db: Session, actor_id: str) -> dict:
    from sqlalchemy import delete, update, text, select
    from app.models.sms_job import SMSJob
    from app.models.checkin import DailyCheckIn
    from app.models.risk_history import RiskClassificationHistory
    from app.models.patient import MedicalHistory, PatientDoctorAssignment, PatientDeletionTicket, PatientReactivationTicket
    from app.models.medication import MedicationOccurrence, MedicationAdherence, MedicationSchedule, Medication
    from app.models.clinical_document import ClinicalDocument, OCRExtraction
    from app.models.recovery_plan import RecoveryPlan
    from app.models.appointment import Appointment
    from app.models.consultation import Consultation, ConsultationMessage, ConsultationParticipant, ConsultationSignal
    from app.models.notification import Notification
    from app.models.chat_session import ChatSession, ChatMessage
    from app.models.audit_log import AuditLog
    from app.models.user import User

    pid = patient.id
    p_code = patient.patient_code
    user_id = patient.user_id

    # 1. SMS jobs (delete completely so no FK violation on patients.id)
    db.execute(delete(SMSJob).where(SMSJob.patient_id == pid))

    # 2. Check-ins & Risk history & Medical history
    db.execute(delete(DailyCheckIn).where(DailyCheckIn.patient_id == pid))
    db.execute(delete(RiskClassificationHistory).where(RiskClassificationHistory.patient_id == pid))
    db.execute(delete(MedicalHistory).where(MedicalHistory.patient_id == pid))

    # 3. Deletion tickets & assignments
    db.execute(delete(PatientDeletionTicket).where(PatientDeletionTicket.patient_id == pid))
    db.execute(delete(PatientReactivationTicket).where(PatientReactivationTicket.patient_id == pid))
    db.execute(delete(PatientDoctorAssignment).where(PatientDoctorAssignment.patient_id == pid))
    try:
        db.execute(text("DELETE FROM patient_staff_assignments WHERE patient_id = :pid"), {"pid": pid})
    except Exception:
        pass

    # 4. Medication hierarchy
    occ_sub = select(MedicationOccurrence.id).where(MedicationOccurrence.patient_id == pid)
    db.execute(delete(MedicationAdherence).where(MedicationAdherence.occurrence_id.in_(occ_sub)))
    db.execute(delete(MedicationOccurrence).where(MedicationOccurrence.patient_id == pid))
    try:
        db.execute(text("DELETE FROM medication_events WHERE patient_id = :pid"), {"pid": pid})
    except Exception:
        pass
    db.execute(delete(MedicationSchedule).where(MedicationSchedule.patient_id == pid))
    db.execute(delete(Medication).where(Medication.patient_id == pid))

    # 5. Documents & OCR extractions
    doc_sub = select(ClinicalDocument.id).where(ClinicalDocument.patient_id == pid)
    db.execute(delete(OCRExtraction).where(OCRExtraction.document_id.in_(doc_sub)))
    db.execute(delete(ClinicalDocument).where(ClinicalDocument.patient_id == pid))
    try:
        db.execute(text("DELETE FROM document_extractions WHERE document_id IN (SELECT id FROM documents WHERE patient_id = :pid)"), {"pid": pid})
        db.execute(text("DELETE FROM documents WHERE patient_id = :pid"), {"pid": pid})
    except Exception:
        pass

    # 6. Recovery alerts & plans & follow ups
    try:
        db.execute(text("DELETE FROM alert_feedback WHERE alert_id IN (SELECT id FROM recovery_alerts WHERE patient_id = :pid)"), {"pid": pid})
        db.execute(text("DELETE FROM recovery_alerts WHERE patient_id = :pid"), {"pid": pid})
        db.execute(text("DELETE FROM recovery_check_ins WHERE patient_id = :pid"), {"pid": pid})
        db.execute(text("DELETE FROM follow_up_appointments WHERE patient_id = :pid"), {"pid": pid})
    except Exception:
        pass
    db.execute(delete(RecoveryPlan).where(RecoveryPlan.patient_id == pid))

    # 7. Consultations & appointments
    c_sub = select(Consultation.id).where(Consultation.patient_id == pid)
    db.execute(delete(ConsultationSignal).where(ConsultationSignal.consultation_id.in_(c_sub)))
    db.execute(delete(ConsultationMessage).where(ConsultationMessage.consultation_id.in_(c_sub)))
    db.execute(delete(ConsultationParticipant).where(ConsultationParticipant.consultation_id.in_(c_sub)))
    db.execute(delete(Consultation).where(Consultation.patient_id == pid))
    db.execute(delete(Appointment).where(Appointment.patient_id == pid))

    # 8. Chat sessions & messages
    if user_id:
        chat_sub = select(ChatSession.id).where((ChatSession.patient_id == pid) | (ChatSession.user_id == user_id))
    else:
        chat_sub = select(ChatSession.id).where(ChatSession.patient_id == pid)
    db.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(chat_sub)))
    if user_id:
        db.execute(delete(ChatSession).where((ChatSession.patient_id == pid) | (ChatSession.user_id == user_id)))
    else:
        db.execute(delete(ChatSession).where(ChatSession.patient_id == pid))

    # 9. Clean up user references if patient had a linked user
    if user_id:
        db.execute(delete(Notification).where(Notification.user_id == user_id))
        try:
            db.execute(text("DELETE FROM password_reset_tickets WHERE requester_user_id = :uid OR resolved_by = :uid"), {"uid": user_id})
        except Exception:
            pass
        db.execute(delete(ConsultationSignal).where(ConsultationSignal.sender_id == user_id))
        db.execute(delete(ConsultationMessage).where(ConsultationMessage.sender_id == user_id))
        db.execute(delete(ConsultationParticipant).where(ConsultationParticipant.user_id == user_id))
        db.execute(update(AuditLog).where(AuditLog.actor_id == user_id).values(actor_id=None))

    # 10. Delete Patient
    db.delete(patient)
    db.flush()

    # 11. Delete associated Patient User
    if user_id:
        u = db.get(User, user_id)
        if u and u.role == Role.PATIENT.value:
            db.delete(u)
            db.flush()

    record_audit(
        db=db,
        actor_id=actor_id,
        action="patient_deleted",
        target_type="patient",
        target_id=pid,
        details={"patient_code": p_code}
    )
    db.commit()
    return {"message": f"Patient {p_code} and all associated records have been completely deleted.", "patient_id": pid}


@router.get("/deletion-requests")
def list_deletion_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    query = select(PatientDeletionTicket).order_by(
        PatientDeletionTicket.status == "pending",
        PatientDeletionTicket.created_at.desc()
    )
    if current_user.role != Role.ADMIN.value:
        query = query.where(PatientDeletionTicket.requested_by == current_user.id)

    tickets = db.scalars(query).all()
    results = []
    for t in tickets:
        pat = db.get(Patient, t.patient_id)
        pat_user = db.get(User, pat.user_id) if pat and pat.user_id else None
        requester = db.get(User, t.requested_by)
        reviewer = db.get(User, t.reviewed_by) if t.reviewed_by else None

        results.append({
            "id": t.id,
            "patient_id": t.patient_id,
            "patient_code": pat.patient_code if pat else "N/A",
            "patient_name": pat_user.name if pat_user else (pat.patient_code if pat else "Unknown"),
            "requested_by": t.requested_by,
            "requester_name": requester.name if requester else "Staff",
            "requester_role": requester.role if requester else "staff",
            "reason": t.reason,
            "status": t.status,
            "admin_notes": t.admin_notes,
            "reviewed_by": t.reviewed_by,
            "reviewer_name": reviewer.name if reviewer else None,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "reviewed_at": t.reviewed_at.isoformat() if t.reviewed_at else None,
        })
    return results


@router.post("/deletion-requests/{ticket_id}/approve")
def approve_deletion_request(
    ticket_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    ticket = db.get(PatientDeletionTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Deletion ticket not found.")
    if ticket.status != "pending":
        raise HTTPException(status_code=400, detail=f"Ticket is already {ticket.status}.")

    patient = db.get(Patient, ticket.patient_id)
    if not patient:
        ticket.status = "approved"
        ticket.reviewed_by = current_user.id
        ticket.reviewed_at = dt.datetime.utcnow()
        db.commit()
        return {"message": "Patient was already deleted. Ticket marked approved.", "ticket_id": ticket_id}

    p_code = patient.patient_code

    # Perform cascading deletion
    _perform_patient_deletion(patient=patient, db=db, actor_id=current_user.id)

    ticket.status = "approved"
    ticket.reviewed_by = current_user.id
    ticket.reviewed_at = dt.datetime.utcnow()

    # Send in-app notification to requester
    from app.services.notification_service import create_in_app_notification
    create_in_app_notification(
        db=db,
        user_id=ticket.requested_by,
        title="Patient Deletion Approved",
        message=f"Administrator {current_user.name} approved your deletion request for patient {p_code}. The patient has been permanently deleted.",
        category="deletion_request"
    )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="deletion_ticket_approved",
        target_type="patient_deletion_ticket",
        target_id=ticket.id,
        details={"patient_code": p_code}
    )
    db.commit()
    return {"message": f"Deletion ticket approved and patient {p_code} deleted successfully.", "ticket_id": ticket_id}


@router.post("/deletion-requests/{ticket_id}/deny")
def deny_deletion_request(
    ticket_id: str,
    req: PatientDeletionTicketReviewIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    ticket = db.get(PatientDeletionTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Deletion ticket not found.")
    if ticket.status != "pending":
        raise HTTPException(status_code=400, detail=f"Ticket is already {ticket.status}.")

    ticket.status = "rejected"
    ticket.admin_notes = req.admin_notes or ""
    ticket.reviewed_by = current_user.id
    ticket.reviewed_at = dt.datetime.utcnow()

    patient = db.get(Patient, ticket.patient_id)
    p_code = patient.patient_code if patient else "patient"

    # Send in-app notification to requester
    from app.services.notification_service import create_in_app_notification
    create_in_app_notification(
        db=db,
        user_id=ticket.requested_by,
        title="Patient Deletion Denied",
        message=f"Administrator {current_user.name} denied your deletion request for {p_code}. Note: {req.admin_notes or 'No reason provided.'}",
        category="deletion_request"
    )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="deletion_ticket_denied",
        target_type="patient_deletion_ticket",
        target_id=ticket.id,
        details={"admin_notes": req.admin_notes}
    )
    db.commit()
    return {"message": "Deletion request has been denied.", "ticket_id": ticket_id}


@router.get("/reactivation-requests")
def list_reactivation_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    query = select(PatientReactivationTicket).order_by(
        PatientReactivationTicket.status == "pending",
        PatientReactivationTicket.created_at.desc()
    )
    tickets = db.scalars(query).all()
    results = []
    for t in tickets:
        pat = db.get(Patient, t.patient_id)
        pat_user = db.get(User, pat.user_id) if pat and pat.user_id else None
        requester = db.get(User, t.requested_by)
        reviewer = db.get(User, t.reviewed_by) if t.reviewed_by else None

        results.append({
            "id": t.id,
            "patient_id": t.patient_id,
            "patient_code": pat.patient_code if pat else "N/A",
            "patient_name": pat_user.name if pat_user else (pat.patient_code if pat else "Unknown"),
            "patient_phone": pat.phone if pat else None,
            "requested_by": t.requested_by,
            "requester_name": requester.name if requester else "Patient",
            "reason": t.reason,
            "status": t.status,
            "clinical_notes": t.clinical_notes,
            "reviewed_by": t.reviewed_by,
            "reviewer_name": reviewer.name if reviewer else None,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "reviewed_at": t.reviewed_at.isoformat() if t.reviewed_at else None,
        })
    return results


@router.post("/reactivation-requests/{ticket_id}/approve")
def approve_reactivation_request(
    ticket_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    ticket = db.get(PatientReactivationTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Reactivation request not found.")

    if ticket.status != "pending":
        raise HTTPException(status_code=400, detail=f"Ticket is already {ticket.status}.")

    ticket.status = "approved"
    ticket.reviewed_by = current_user.id
    ticket.reviewed_at = dt.datetime.utcnow()

    patient = db.get(Patient, ticket.patient_id)
    if patient:
        patient.care_status = "active"
        patient.updated_at = dt.datetime.utcnow()

        from app.services.notification_service import create_in_app_notification
        if patient.user_id:
            create_in_app_notification(
                db=db,
                user_id=patient.user_id,
                title="Care Protocol Reactivated",
                message=f"Clinical staff {current_user.name} ({current_user.role}) approved your reactivation request. Your care protocol is now Active.",
                category="care_status",
                link="#/patient/overview"
            )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="reactivation_ticket_approved",
        target_type="patient_reactivation_ticket",
        target_id=ticket.id,
        details={"patient_id": ticket.patient_id}
    )
    db.commit()
    return {"message": "Reactivation request approved. Patient care status is now Active.", "ticket_id": ticket_id}


@router.post("/reactivation-requests/{ticket_id}/reject")
def reject_reactivation_request(
    ticket_id: str,
    req: ReactivationTicketReviewIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    ticket = db.get(PatientReactivationTicket, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Reactivation request not found.")

    if ticket.status != "pending":
        raise HTTPException(status_code=400, detail=f"Ticket is already {ticket.status}.")

    ticket.status = "rejected"
    ticket.clinical_notes = req.clinical_notes or ""
    ticket.reviewed_by = current_user.id
    ticket.reviewed_at = dt.datetime.utcnow()

    patient = db.get(Patient, ticket.patient_id)
    if patient and patient.user_id:
        from app.services.notification_service import create_in_app_notification
        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title="Care Reactivation Request Update",
            message=f"Clinical staff reviewed your reactivation request: {req.clinical_notes or 'Your condition remains stable under current monitoring.'}",
            category="care_status",
            link="#/patient/overview"
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="reactivation_ticket_rejected",
        target_type="patient_reactivation_ticket",
        target_id=ticket.id,
        details={"clinical_notes": req.clinical_notes}
    )
    db.commit()
    return {"message": "Reactivation request has been rejected.", "ticket_id": ticket_id}


@router.patch("/{pid}/care-status")
def update_patient_care_status(
    pid: str,
    req: CareStatusUpdateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    status_val = req.care_status.strip().lower()
    if status_val not in ("active", "idle"):
        raise HTTPException(status_code=400, detail="Care status must be 'active' or 'idle'.")

    patient.care_status = status_val
    patient.updated_at = dt.datetime.utcnow()

    from app.services.notification_service import create_in_app_notification
    if patient.user_id:
        if status_val == "idle":
            title = "Care Status Set to Idle / Stable"
            msg = f"Your clinical team has transitioned your care status to Idle (treatment stable). Daily check-in alerts are paused. Note: {req.reason or 'Patient condition evaluated as stable.'}"
        else:
            title = "Care Status Set to Active"
            msg = f"Your clinical team has reactivated your active recovery care protocol. Note: {req.reason or 'Care protocol active.'}"

        create_in_app_notification(
            db=db,
            user_id=patient.user_id,
            title=title,
            message=msg,
            category="care_status",
            link="#/patient/overview"
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_care_status_updated",
        target_type="patient",
        target_id=pid,
        details={"care_status": status_val, "reason": req.reason}
    )
    db.commit()
    return patient_to_dict(patient, db)


@router.patch("/{pid}/condition")
@router.put("/{pid}/condition")
def update_patient_condition(
    pid: str,
    req: PatientConditionUpdateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    """Clinical endpoint for doctors/nurses to update patient's active disease/condition and stage."""
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    patient.active_condition = req.condition.strip() if req.condition else None
    patient.condition_stage = req.stage or "recovering"
    patient.condition_notes = req.notes.strip() if req.notes else ""
    patient.condition_updated_at = dt.datetime.utcnow()
    patient.updated_at = dt.datetime.utcnow()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_condition_updated",
        target_type="patient",
        target_id=patient.id,
        details={
            "condition": patient.active_condition,
            "stage": patient.condition_stage,
            "notes": patient.condition_notes
        }
    )
    db.commit()

    # Re-evaluate patient risk with updated condition
    try:
        from app.services.risk_service import evaluate_patient_risk
        evaluate_patient_risk(db, patient_id=patient.id, window_days=7, force=True)
        db.commit()
    except Exception:
        pass

    return patient_to_dict(patient, db)


@router.post("/{pid}/reactivation-request")
def request_care_reactivation(
    pid: str,
    req: ReactivationTicketCreateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # Check for existing pending request
    existing = db.scalar(
        select(PatientReactivationTicket).where(
            PatientReactivationTicket.patient_id == pid,
            PatientReactivationTicket.status == "pending"
        )
    )
    if existing:
        return {
            "message": "A care reactivation request is already pending review with your clinical team.",
            "ticket_id": existing.id,
            "status": "pending"
        }

    ticket = PatientReactivationTicket(
        patient_id=pid,
        requested_by=current_user.id,
        reason=req.reason.strip(),
        status="pending",
        created_at=dt.datetime.utcnow()
    )
    db.add(ticket)
    db.flush()

    pat_name = patient.user.name if patient.user else patient.patient_code
    # Notify clinical staff & admins
    from app.services.notification_service import create_in_app_notification
    staff_targets = db.scalars(
        select(User).where(User.role.in_([Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value]), User.active == True)
    ).all()
    for s in staff_targets:
        create_in_app_notification(
            db=db,
            user_id=s.id,
            title="Patient Care Reactivation Request",
            message=f"Patient {pat_name} ({patient.patient_code}) requested Care Reactivation. Reason: \"{req.reason.strip()}\"",
            category="care_status",
            link="#/staff/patients"
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="care_reactivation_requested",
        target_type="patient",
        target_id=pid,
        details={"reason": req.reason.strip(), "ticket_id": ticket.id}
    )
    db.commit()
    return {
        "message": "Reactivation request submitted successfully to your clinical team.",
        "ticket_id": ticket.id,
        "status": "pending"
    }


@router.get("/{pid}")
def get_patient(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)
    return patient_to_dict(patient, db)

@router.put("/{pid}")
def update_patient(
    pid: str,
    req: PatientUpdateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    # Doctor access guard: doctors can ONLY edit patients they have been assigned to
    if current_user.role == Role.DOCTOR.value:
        if not is_doctor_assigned_to_patient(current_user.id, pid, db):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Doctors can only view and edit patients they have been assigned to."
            )

    user = patient.user if patient.user else db.get(User, patient.user_id) if patient.user_id else None

    if req.active is not None:
        patient.active = req.active
        if user:
            user.active = req.active

    if req.medical_history is not None:
        med_hist = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == pid))
        if not med_hist:
            med_hist = MedicalHistory(patient_id=pid)
            db.add(med_hist)
        med_hist.has_history = req.medical_history.has_history
        med_hist.conditions = req.medical_history.conditions
        med_hist.past_medications = req.medical_history.past_medications
        med_hist.past_treatments = req.medical_history.past_treatments
        med_hist.past_hospitalizations = req.medical_history.past_hospitalizations
        med_hist.notes = req.medical_history.notes or ""
        med_hist.updated_at = dt.datetime.utcnow()
        patient.previous_history = req.medical_history.has_history

    if req.name and user:
        user.name = req.name.strip()

    # Handle Age vs DOB
    if req.age is not None and req.age > 0 and req.dob is None:
        today = dt.date.today()
        patient.dob = dt.date(today.year - req.age, today.month, min(today.day, 28))
    elif req.dob is not None:
        patient.dob = req.dob

    if req.gender is not None:
        patient.gender = req.gender
    if req.phone is not None:
        patient.phone = normalize_phone(req.phone)
    if req.email and user:
        user.email = req.email.strip().lower()
    if req.preferred_language is not None:
        patient.preferred_language = req.preferred_language
    if getattr(req, "sms_language", None) is not None:
        patient.sms_language = req.sms_language
    if req.sms_opt_in is not None:
        patient.sms_opt_in = req.sms_opt_in
        if req.sms_opt_in:
            patient.sms_opt_in_at = dt.datetime.utcnow()
            patient.sms_opt_in_recorded_by = current_user.id

    # If nurse or admin supplies assigned_doctor_ids, sync multi-doctor assignments
    if req.assigned_doctor_ids is not None and current_user.role in (Role.ADMIN.value, Role.NURSE.value):
        from app.services.notification_service import create_in_app_notification
        existing_assignments = db.scalars(
            select(PatientDoctorAssignment).where(
                PatientDoctorAssignment.patient_id == pid,
                PatientDoctorAssignment.is_active == True
            )
        ).all()
        existing_doc_ids = {a.doctor_id for a in existing_assignments}
        target_doc_ids = set(req.assigned_doctor_ids)

        # Revoke removed doctors
        for a in existing_assignments:
            if a.doctor_id not in target_doc_ids:
                a.is_active = False
                a.revoked_at = dt.datetime.utcnow()
                a.revoked_by = current_user.id

        # Add newly added doctors
        for doc_id in target_doc_ids:
            if doc_id not in existing_doc_ids:
                doc = db.get(User, doc_id)
                if doc and doc.role == Role.DOCTOR.value:
                    db.add(PatientDoctorAssignment(
                        patient_id=pid,
                        doctor_id=doc.id,
                        assigned_by=current_user.id,
                        department=doc.specialization,
                        is_active=True,
                        assigned_at=dt.datetime.utcnow()
                    ))
                    pat_user = db.get(User, patient.user_id) if patient.user_id else None
                    patient_display_name = pat_user.name if pat_user else patient.name
                    create_in_app_notification(
                        db=db,
                        user_id=doc.id,
                        title="New Patient Assigned",
                        message=f"Patient {patient_display_name} has been assigned to you by {current_user.name}.",
                        category="assignment",
                        link=f"#/staff/patients/{patient.id}/overview"
                    )

        patient.assigned_doctor_id = req.assigned_doctor_ids[0] if req.assigned_doctor_ids else None

    patient.updated_at = dt.datetime.utcnow()
    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_updated",
        target_type="patient",
        target_id=patient.id,
        details={"updated_fields": list(req.model_dump(exclude_unset=True).keys())}
    )
    db.commit()
    return patient_to_dict(patient, db)


@router.put("/{pid}/preferences")
def update_patient_preferences(
    pid: str,
    req: PatientPreferencesIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)

    if req.preferred_language is not None:
        lang = req.preferred_language.strip().lower()
        if lang not in ("en", "ta", "hi"):
            raise HTTPException(status_code=400, detail="Invalid language. Supported: en, ta, hi")
        patient.preferred_language = lang

    if req.sms_language is not None:
        sms_l = req.sms_language.strip().lower()
        if sms_l not in ("en", "ta", "hi"):
            raise HTTPException(status_code=400, detail="Invalid SMS language. Supported: en, ta, hi")
        patient.sms_language = sms_l

    if req.sms_opt_in is not None:
        if req.sms_opt_in and not patient.phone:
            raise HTTPException(status_code=400, detail="Cannot enable SMS without a registered phone number.")
        patient.sms_opt_in = req.sms_opt_in
        if req.sms_opt_in:
            patient.sms_opt_in_at = dt.datetime.utcnow()
            patient.sms_opt_in_recorded_by = current_user.id

    patient.updated_at = dt.datetime.utcnow()
    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_preferences_updated",
        target_type="patient",
        target_id=patient.id,
        details={"preferred_language": patient.preferred_language, "sms_opt_in": patient.sms_opt_in}
    )
    db.commit()
    return patient_to_dict(patient, db)


@router.post("/{pid}/assign-doctor")
def assign_doctor_endpoint(
    pid: str,
    req: AssignDoctorRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")
    doctor = db.get(User, req.doctor_id)
    if not doctor or doctor.role != Role.DOCTOR.value:
        raise HTTPException(status_code=400, detail="Invalid doctor selected.")

    existing = db.scalar(
        select(PatientDoctorAssignment).where(
            PatientDoctorAssignment.patient_id == pid,
            PatientDoctorAssignment.doctor_id == req.doctor_id,
            PatientDoctorAssignment.is_active == True
        )
    )
    if not existing:
        db.add(PatientDoctorAssignment(
            patient_id=pid,
            doctor_id=req.doctor_id,
            assigned_by=current_user.id,
            department=req.department or doctor.specialization,
            notes=req.notes,
            is_active=True,
            assigned_at=dt.datetime.utcnow()
        ))
        if not patient.assigned_doctor_id:
            patient.assigned_doctor_id = req.doctor_id
        db.flush()

        from app.services.notification_service import create_in_app_notification
        pat_name = patient.user.name if patient.user else patient.patient_code
        create_in_app_notification(
            db=db,
            user_id=doctor.id,
            title="New Patient Assigned",
            message=f"Patient {pat_name} ({patient.patient_code}) has been assigned to your clinical care.",
            category="assignment",
            link=f"#/staff/patients/{patient.id}/overview"
        )
        record_audit(
            db=db,
            actor_id=current_user.id,
            action="patient_doctor_assigned",
            target_type="patient",
            target_id=pid,
            new_values={"doctor_id": req.doctor_id, "department": req.department}
        )
        db.commit()

    return patient_to_dict(patient, db)


@router.post("/{pid}/revoke-doctor/{doctor_id}")
def revoke_doctor_endpoint(
    pid: str,
    doctor_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    assignments = db.scalars(
        select(PatientDoctorAssignment).where(
            PatientDoctorAssignment.patient_id == pid,
            PatientDoctorAssignment.doctor_id == doctor_id,
            PatientDoctorAssignment.is_active == True
        )
    ).all()
    for a in assignments:
        a.is_active = False
        a.revoked_at = dt.datetime.utcnow()
        a.revoked_by = current_user.id

    if patient.assigned_doctor_id == doctor_id:
        remaining = db.scalar(
            select(PatientDoctorAssignment.doctor_id).where(
                PatientDoctorAssignment.patient_id == pid,
                PatientDoctorAssignment.is_active == True
            )
        )
        patient.assigned_doctor_id = remaining

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_doctor_revoked",
        target_type="patient",
        target_id=pid,
        details={"doctor_id": doctor_id}
    )
    db.commit()
    return patient_to_dict(patient, db)


@router.post("/{pid}/deletion-request")
def request_patient_deletion(
    pid: str,
    req: PatientDeletionTicketCreateIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    if current_user.role == Role.DOCTOR.value:
        if not is_doctor_assigned_to_patient(current_user.id, pid, db):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Doctors can only request deletion for patients assigned to their care."
            )

    # Check for existing pending request
    existing = db.scalar(
        select(PatientDeletionTicket).where(
            PatientDeletionTicket.patient_id == pid,
            PatientDeletionTicket.status == "pending"
        )
    )
    if existing:
        return {
            "message": "A deletion request is already pending administrator review.",
            "ticket_id": existing.id,
            "status": "pending"
        }

    ticket = PatientDeletionTicket(
        patient_id=pid,
        requested_by=current_user.id,
        reason=req.reason.strip(),
        status="pending",
        created_at=dt.datetime.utcnow()
    )
    db.add(ticket)
    db.flush()

    pat_name = patient.user.name if patient.user else patient.patient_code
    # Notify all administrators with an in-app alert
    from app.services.notification_service import create_in_app_notification
    admins = db.scalars(select(User).where(User.role == Role.ADMIN.value, User.active == True)).all()
    for admin in admins:
        create_in_app_notification(
            db=db,
            user_id=admin.id,
            title="Patient Deletion Request",
            message=f"{current_user.role.capitalize()} {current_user.name} submitted a deletion request for patient {pat_name} ({patient.patient_code}). Reason: {req.reason.strip()}",
            category="deletion_request",
            link="#/admin/deletion-requests"
        )

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_deletion_requested",
        target_type="patient",
        target_id=pid,
        details={"reason": req.reason.strip(), "ticket_id": ticket.id}
    )
    db.commit()

    return {
        "message": "Deletion request submitted successfully to administrators. In-app alerts generated.",
        "ticket_id": ticket.id,
        "status": ticket.status
    }


@router.delete("/{pid}", status_code=status.HTTP_200_OK)
def delete_patient(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    patient = db.get(Patient, pid)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    return _perform_patient_deletion(patient=patient, db=db, actor_id=current_user.id)

@router.get("/{pid}/medical-history")
def get_medical_history(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    patient = enforce_patient_isolation(patient_id=pid, current_user=current_user, db=db)
    history = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient.id))
    if not history or not history.has_history:
        return {
            "has_history": False,
            "display_message": "No previous medical history recorded.",
            "conditions": [],
            "past_medications": [],
            "past_treatments": [],
            "past_hospitalizations": [],
            "notes": ""
        }

    return {
        "has_history": True,
        "conditions": history.conditions or [],
        "past_medications": history.past_medications or [],
        "past_treatments": history.past_treatments or [],
        "past_hospitalizations": history.past_hospitalizations or [],
        "notes": history.notes or "",
        "updated_at": history.updated_at.isoformat() if history.updated_at else None
    }

@router.put("/{pid}/medical-history")
def update_medical_history(
    pid: str,
    req: MedicalHistoryIn,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value, Role.DOCTOR.value, Role.NURSE.value))
):
    patient = db.get(Patient, pid)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    history = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient.id))
    if not history:
        history = MedicalHistory(patient_id=patient.id)
        db.add(history)

    history.has_history = req.has_history
    history.conditions = req.conditions
    history.past_medications = req.past_medications
    history.past_treatments = req.past_treatments
    history.past_hospitalizations = req.past_hospitalizations
    history.notes = req.notes or ""
    history.updated_at = dt.datetime.utcnow()

    patient.previous_history = req.has_history
    patient.updated_at = dt.datetime.utcnow()

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="medical_history_updated",
        target_type="patient",
        target_id=patient.id,
        details={"has_history": req.has_history}
    )
    db.commit()

    return {"message": "Medical history updated successfully."}

@router.post("/{pid}/archive", status_code=status.HTTP_200_OK)
def archive_patient(
    pid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    patient = db.get(Patient, pid)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    patient.active = False
    patient.updated_at = dt.datetime.utcnow()

    # Also deactivate associated user account if present
    user = patient.user if patient.user else db.get(User, patient.user_id) if patient.user_id else None
    if user:
        user.active = False

    record_audit(
        db=db,
        actor_id=current_user.id,
        action="patient_archived",
        target_type="patient",
        target_id=patient.id
    )
    db.commit()

    return {"message": f"Patient {patient.patient_code} has been archived."}
