import datetime as dt
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_

from app.core.database import get_db
from app.core.constants import Role, OccurrenceState, SMSJobStatus
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.appointment import Appointment
from app.models.consultation import Consultation
from app.models.medication import MedicationOccurrence, MedicationAdherence
from app.models.sms_job import SMSJob
from app.models.audit_log import AuditLog
from app.api.deps import require_role

router = APIRouter(prefix="/api/admin", tags=["Hospital Administration & Analytics"])


@router.get("/dashboard")
def get_admin_dashboard_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role(Role.ADMIN.value))
):
    """
    Comprehensive administrative analytics, clinical KPIs, departmental distributions,
    and system operational metrics for the Admin Dashboard.
    """
    now = dt.datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

    # 1. User & Staff Counts
    total_patients = db.scalar(select(func.count(Patient.id)).where(Patient.active == True)) or 0
    total_doctors = db.scalar(select(func.count(User.id)).where(User.role == Role.DOCTOR.value, User.active == True)) or 0
    total_nurses = db.scalar(select(func.count(User.id)).where(User.role == Role.NURSE.value, User.active == True)) or 0

    # 2. Appointment Metrics
    now_utc = dt.datetime.utcnow()
    total_appointments = db.scalar(select(func.count(Appointment.id))) or 0
    today_appointments = db.scalar(
        select(func.count(Appointment.id)).where(
            Appointment.start_time >= today_start,
            Appointment.start_time <= today_end
        )
    ) or 0
    upcoming_appointments = db.scalar(
        select(func.count(Appointment.id)).where(
            Appointment.status.in_(["scheduled", "rescheduled"]),
            Appointment.start_time >= (now_utc - dt.timedelta(minutes=30))
        )
    ) or 0
    completed_appointments = db.scalar(select(func.count(Appointment.id)).where(Appointment.status == "completed")) or 0
    reschedule_requests = db.scalar(select(func.count(Appointment.id)).where(Appointment.decision_status == "reschedule_requested")) or 0
    missed_appointments = db.scalar(select(func.count(Appointment.id)).where(Appointment.status == "missed")) or 0

    # 3. Telemedicine / Consultations
    active_consultations = db.scalar(
        select(func.count(Consultation.id)).where(Consultation.status.in_(["waiting", "active"]))
    ) or 0
    total_consultations = db.scalar(select(func.count(Consultation.id))) or 0

    # 4. Medication Adherence Rate (Evaluated on elapsed/due occurrences up to now)
    due_occurrences_count = db.scalar(
        select(func.count(MedicationOccurrence.id)).where(
            MedicationOccurrence.scheduled_at <= now,
            MedicationOccurrence.state != OccurrenceState.CANCELLED.value
        )
    ) or 0
    taken_occurrences = db.scalar(
        select(func.count(MedicationOccurrence.id)).where(
            MedicationOccurrence.scheduled_at <= now,
            MedicationOccurrence.state == OccurrenceState.TAKEN.value
        )
    ) or 0
    adherence_rate = round((taken_occurrences / due_occurrences_count * 100), 1) if due_occurrences_count > 0 else 100.0

    # 5. SMS Gateway Stats
    total_sms_sent = db.scalar(
        select(func.count(SMSJob.id)).where(SMSJob.state.in_([SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value]))
    ) or 0
    total_sms_failed = db.scalar(select(func.count(SMSJob.id)).where(SMSJob.state == SMSJobStatus.FAILED.value)) or 0
    total_sms_queued = db.scalar(select(func.count(SMSJob.id)).where(SMSJob.state == SMSJobStatus.QUEUED.value)) or 0
    today_sms_sent = db.scalar(
        select(func.count(SMSJob.id)).where(
            SMSJob.created_at >= today_start,
            SMSJob.state.in_([SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value])
        )
    ) or 0
    sms_attempted = total_sms_sent + total_sms_failed
    sms_delivery_rate = round((total_sms_sent / sms_attempted * 100), 1) if sms_attempted > 0 else 0.0

    # 6. Precompute Doctor Workloads in 2 Grouped Queries (Avoid O(N) DB roundtrips)
    multi_counts = dict(
        db.execute(
            select(PatientDoctorAssignment.doctor_id, func.count(PatientDoctorAssignment.id))
            .where(PatientDoctorAssignment.is_active == True)
            .group_by(PatientDoctorAssignment.doctor_id)
        ).all()
    )
    direct_counts = dict(
        db.execute(
            select(Patient.assigned_doctor_id, func.count(Patient.id))
            .where(Patient.active == True, Patient.assigned_doctor_id != None)
            .group_by(Patient.assigned_doctor_id)
        ).all()
    )

    specialty_counts: Dict[str, int] = {}
    doctors_list = []
    doctors = db.scalars(select(User).where(User.role == Role.DOCTOR.value, User.active == True)).all()
    for d in doctors:
        assigned_count = multi_counts.get(d.id, 0)
        direct_pt_count = direct_counts.get(d.id, 0)
        caseload = max(assigned_count, direct_pt_count)

        spec = (d.specialization or "general_medicine").replace("_", " ").title()
        specialty_counts[spec] = specialty_counts.get(spec, 0) + caseload

        doctors_list.append({
            "id": d.id,
            "name": d.name,
            "email": d.email,
            "phone": d.phone,
            "specialization": d.specialization,
            "availability_status": d.availability_status or "online",
            "away_until": d.away_until.isoformat() if d.away_until else None,
            "active_patients_count": caseload,
            "created_at": d.created_at.isoformat() if d.created_at else None
        })

    if not specialty_counts:
        specialty_counts = {"General Medicine": total_patients}
    else:
        unassigned_count = max(0, total_patients - sum(specialty_counts.values()))
        if unassigned_count > 0:
            specialty_counts["Unassigned (Triage)"] = unassigned_count

    department_chart = [
        {"department": dept, "patient_count": count}
        for dept, count in sorted(specialty_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    # 7. Doctor Status Distribution
    doctor_status_distribution = {
        "online": sum(1 for d in doctors if (d.availability_status or "online") == "online"),
        "away": sum(1 for d in doctors if (d.availability_status or "") == "away"),
        "busy": sum(1 for d in doctors if (d.availability_status or "") == "busy"),
    }

    # 8. SMS Categories Breakdown
    sms_categories: Dict[str, int] = {}
    jobs = db.scalars(select(SMSJob)).all()
    for j in jobs:
        cat = j.category or "general"
        sms_categories[cat] = sms_categories.get(cat, 0) + 1

    # 9. Detailed Nurses List
    nurses = db.scalars(select(User).where(User.role == Role.NURSE.value, User.active == True)).all()
    nurses_list = [
        {
            "id": n.id,
            "name": n.name,
            "email": n.email,
            "phone": n.phone,
            "created_at": n.created_at.isoformat() if n.created_at else None
        }
        for n in nurses
    ]

    # 10. Recent Audit Logs (Batched Actor Lookup)
    audits = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(15)).all()
    actor_ids = {a.actor_id for a in audits if a.actor_id}
    actors_map = {u.id: u for u in db.scalars(select(User).where(User.id.in_(actor_ids))).all()} if actor_ids else {}

    audit_trail = []
    for a in audits:
        actor = actors_map.get(a.actor_id) if a.actor_id else None
        audit_trail.append({
            "id": a.id,
            "action": a.action,
            "actor_name": actor.name if actor else (a.actor_id or "System"),
            "target_type": a.target_type,
            "target_id": a.target_id,
            "details": a.details or a.new_values,
            "created_at": a.created_at.isoformat() if a.created_at else None
        })

    outcomes_dict = {
        "completed": completed_appointments,
        "scheduled": upcoming_appointments if upcoming_appointments > 0 else max(0, total_appointments - completed_appointments - missed_appointments),
        "reschedule_requested": reschedule_requests,
        "missed": missed_appointments
    }

    sms_health_dict = {
        "delivery_rate_pct": sms_delivery_rate,
        "total_sent": total_sms_sent,
        "today_sent": today_sms_sent,
        "total_queued": total_sms_queued,
        "total_failed": total_sms_failed
    }

    return {
        "summary": {
            "total_patients": total_patients,
            "total_doctors": total_doctors,
            "total_nurses": total_nurses,
            "total_appointments": total_appointments,
            "completed_appointments": completed_appointments,
            "today_appointments": today_appointments,
            "upcoming_appointments": upcoming_appointments,
            "scheduled_appointments": upcoming_appointments,
            "active_consultations": active_consultations,
            "total_consultations": total_consultations,
            "adherence_rate": adherence_rate,
            "adherence_rate_pct": adherence_rate,
            "sms_delivery_rate_pct": sms_delivery_rate,
            "total_sms_sent": total_sms_sent,
            "today_sms_sent": today_sms_sent,
            "total_sms_queued": total_sms_queued,
            "total_sms_failed": total_sms_failed,
        },
        "department_chart": department_chart,
        "appointment_outcomes": outcomes_dict,
        "sms_health": sms_health_dict,
        "charts": {
            "department_distribution": department_chart,
            "doctor_status_distribution": doctor_status_distribution,
            "appointment_outcomes": outcomes_dict,
            "sms_category_breakdown": sms_categories
        },
        "doctors": doctors_list,
        "nurses": nurses_list,
        "audit_trail": audit_trail
    }
