"""
Recovera AI - Dynamic Clinician Roster & Auto-Load Balancing Service (Phase 14)

Calculates live clinician workload metrics and provides intelligent load-balanced
assignment matching clinical specialty and minimizing clinician burnout.
"""
import datetime as dt
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func

from app.core.constants import Role, AppointmentStatus
from app.core.audit import record_audit
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.appointment import Appointment
from app.services.notification_service import notify_doctor_patient_assigned

def calculate_doctor_workload(doctor: User, db: Session) -> Dict[str, Any]:
    """Computes real-time workload metrics for an individual doctor."""
    # Count assigned active patients (both direct and multi-doctor active assignments)
    primary_patient_ids = set(
        db.scalars(
            select(Patient.id).where(
                Patient.assigned_doctor_id == doctor.id,
                Patient.active == True
            )
        ).all()
    )
    assigned_records = set(
        db.scalars(
            select(PatientDoctorAssignment.patient_id).where(
                PatientDoctorAssignment.doctor_id == doctor.id,
                PatientDoctorAssignment.is_active == True
            )
        ).all()
    )
    total_active_patients = len(primary_patient_ids.union(assigned_records))

    # Appointments today
    today = dt.date.today()
    start_of_day = dt.datetime.combine(today, dt.time.min)
    end_of_day = dt.datetime.combine(today, dt.time.max)

    todays_active_appts = db.scalar(
        select(func.count(Appointment.id)).where(
            Appointment.doctor_id == doctor.id,
            Appointment.start_time >= start_of_day,
            Appointment.start_time <= end_of_day,
            Appointment.status.in_(["pending", "confirmed", "started"])
        )
    ) or 0

    todays_completed_appts = db.scalar(
        select(func.count(Appointment.id)).where(
            Appointment.doctor_id == doctor.id,
            Appointment.start_time >= start_of_day,
            Appointment.start_time <= end_of_day,
            Appointment.status == "completed"
        )
    ) or 0

    # Calculate workload index
    # Load score = (active_patients * 1.5) + (active_appointments_today * 2.5) + status_penalty
    status_penalty = 0
    if doctor.availability_status == "away":
        status_penalty = 500
    elif doctor.availability_status == "busy":
        status_penalty = 15

    load_score = round((total_active_patients * 1.5) + (todays_active_appts * 2.5) + status_penalty, 2)

    return {
        "doctor_id": doctor.id,
        "doctor_name": doctor.name,
        "email": doctor.email,
        "phone": doctor.phone,
        "specialization": doctor.specialization or "general_medicine",
        "availability_status": doctor.availability_status or "online",
        "away_until": doctor.away_until.isoformat() if doctor.away_until else None,
        "active_patients_count": total_active_patients,
        "todays_appointments_count": todays_active_appts,
        "todays_completed_count": todays_completed_appts,
        "load_score": load_score,
        "is_available": doctor.availability_status == "online" and doctor.active
    }

def get_doctor_roster_stats(db: Session) -> List[Dict[str, Any]]:
    """Retrieves live workload statistics across all active doctors in the hospital."""
    doctors = db.scalars(
        select(User).where(
            User.role == Role.DOCTOR.value,
            User.active == True
        )
    ).all()

    stats = [calculate_doctor_workload(doc, db) for doc in doctors]
    # Sort primarily by lowest load_score
    stats.sort(key=lambda s: s["load_score"])
    return stats

def auto_assign_best_doctor(
    patient_id: str,
    db: Session,
    assigned_by_user_id: str,
    specialization: Optional[str] = None,
    department: Optional[str] = None
) -> Dict[str, Any]:
    """
    Intelligently assigns the least-loaded qualified clinician to the patient.
    Matches required specialization if provided, falling back to general medicine.
    """
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found or inactive.")

    # Fetch all active doctors
    doctors = db.scalars(
        select(User).where(
            User.role == Role.DOCTOR.value,
            User.active == True
        )
    ).all()

    if not doctors:
        raise HTTPException(status_code=404, detail="No active clinicians available in hospital roster.")

    # Filter by specialization if requested
    eligible_doctors = doctors
    if specialization and specialization.strip():
        spec_clean = specialization.strip().lower()
        matched = [
            d for d in doctors
            if d.specialization and spec_clean in d.specialization.lower()
        ]
        if matched:
            eligible_doctors = matched

    # Calculate workload for each eligible doctor
    workloads = [calculate_doctor_workload(d, db) for d in eligible_doctors]

    # Prioritize:
    # 1. Availability (online > busy > away)
    # 2. Lowest load_score
    def sort_key(w):
        status_rank = 0 if w["availability_status"] == "online" else (1 if w["availability_status"] == "busy" else 2)
        return (status_rank, w["load_score"])

    workloads.sort(key=sort_key)
    selected_workload = workloads[0]
    selected_doctor_id = selected_workload["doctor_id"]
    selected_doctor = db.get(User, selected_doctor_id)

    # Perform assignment
    old_doctor_id = patient.assigned_doctor_id
    patient.assigned_doctor_id = selected_doctor_id
    patient.updated_at = dt.datetime.utcnow()

    # Multi-doctor assignment record
    # Check if already active
    existing_assign = db.scalar(
        select(PatientDoctorAssignment).where(
            PatientDoctorAssignment.patient_id == patient_id,
            PatientDoctorAssignment.doctor_id == selected_doctor_id,
            PatientDoctorAssignment.is_active == True
        )
    )
    if not existing_assign:
        assignment_record = PatientDoctorAssignment(
            patient_id=patient_id,
            doctor_id=selected_doctor_id,
            assigned_by=assigned_by_user_id,
            department=department or selected_doctor.specialization or "General Medicine",
            notes=f"Auto-assigned via dynamic load balancing (Load score: {selected_workload['load_score']}).",
            is_active=True
        )
        db.add(assignment_record)

    record_audit(
        db=db,
        actor_id=assigned_by_user_id,
        action="patient_auto_assigned_doctor",
        target_type="patient",
        target_id=patient.id,
        details={
            "old_doctor_id": old_doctor_id,
            "new_doctor_id": selected_doctor.id,
            "doctor_name": selected_doctor.name,
            "specialization": selected_doctor.specialization,
            "load_score": selected_workload["load_score"],
            "active_patients_count": selected_workload["active_patients_count"]
        }
    )

    notify_doctor_patient_assigned(db=db, doctor_id=selected_doctor.id, patient=patient)
    db.commit()

    return {
        "message": f"Patient successfully load-balanced and assigned to Dr. {selected_doctor.name}.",
        "patient_id": patient.id,
        "assigned_doctor": selected_workload,
        "specialization_requested": specialization,
        "total_eligible_doctors_evaluated": len(eligible_doctors)
    }
