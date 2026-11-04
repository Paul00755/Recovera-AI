import datetime as dt
import calendar
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func, and_, or_

from app.core.database import get_db
from app.core.constants import Role, OccurrenceState, SMSJobStatus, DocStatus, RiskLevel
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment, PatientDeletionTicket
from app.models.appointment import Appointment
from app.models.consultation import Consultation
from app.models.clinical_document import ClinicalDocument
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence, MedicationAdherence
from app.models.checkin import DailyCheckIn
from app.models.sms_job import SMSJob
from app.models.audit_log import AuditLog
from app.models.risk_history import RiskClassificationHistory
from app.api.deps import get_current_user

router = APIRouter(prefix="/api/analytics", tags=["Clinical Workflow Analytics"])


def compute_date_range(timeline: str, start_date_str: Optional[str] = None, end_date_str: Optional[str] = None):
    """
    Computes (start_dt, end_dt, window_label, granularity) based on requested timeline.
    Granularity can be 'hour' (for single days) or 'day' (for multi-day periods).
    """
    now = dt.datetime.utcnow()
    local_today = dt.date.today()
    max_today = max(now.date(), local_today)
    min_today = min(now.date(), local_today)
    today_start = dt.datetime.combine(min_today, dt.time.min)
    today_end = dt.datetime.combine(max_today, dt.time.max)

    if timeline == "yesterday":
        y_start = today_start - dt.timedelta(days=1)
        y_end = today_end - dt.timedelta(days=1)
        return y_start, y_end, f"Yesterday ({y_start.strftime('%d %b %Y')})", "hour"

    elif timeline == "this_week":
        # Past 7 days
        w_start = today_start - dt.timedelta(days=6)
        return w_start, today_end, f"Past 7 Days ({w_start.strftime('%d %b')} – {today_end.strftime('%d %b %Y')})", "day"

    elif timeline == "this_month":
        m_start = today_start.replace(day=1)
        return m_start, today_end, f"This Month ({now.strftime('%B %Y')})", "day"

    elif timeline == "last_month":
        first_this_month = today_start.replace(day=1)
        last_day_prev_month = first_this_month - dt.timedelta(days=1)
        start_prev_month = last_day_prev_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_prev_month = last_day_prev_month.replace(hour=23, minute=59, second=59, microsecond=999999)
        return start_prev_month, end_prev_month, f"Last Month ({last_day_prev_month.strftime('%B %Y')})", "day"

    elif timeline == "custom" and start_date_str:
        try:
            s_parsed = dt.datetime.fromisoformat(start_date_str.split("T")[0])
            s_dt = s_parsed.replace(hour=0, minute=0, second=0, microsecond=0)
            if end_date_str:
                e_parsed = dt.datetime.fromisoformat(end_date_str.split("T")[0])
                e_dt = e_parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
            else:
                e_dt = s_parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
            diff_days = (e_dt - s_dt).days
            gran = "hour" if diff_days <= 1 else "day"
            label = s_dt.strftime('%d %b %Y') if diff_days <= 1 else f"{s_dt.strftime('%d %b %Y')} – {e_dt.strftime('%d %b %Y')}"
            return s_dt, e_dt, f"Custom Date ({label})", gran
        except Exception:
            return today_start, today_end, f"Today ({today_start.strftime('%d %b %Y')})", "hour"

    else:  # default "today"
        return today_start, today_end, f"Today ({today_start.strftime('%d %b %Y')})", "hour"


@router.get("/workflow")
def get_workflow_analytics(
    timeline: str = Query("today", description="today, yesterday, this_week, this_month, last_month, custom"),
    start_date: Optional[str] = Query(None, description="Start date YYYY-MM-DD for custom date"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD for custom date"),
    patient_id: Optional[str] = Query(None, description="Optional target patient ID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Unified, role-aware clinical analytics grounded directly in actual database records.
    Provides timeline switching (today, yesterday, specific-date, this-month, last-month),
    SVG time-series buckets, department and triage distributions, and executive KPIs.
    """
    start_dt, end_dt, window_label, granularity = compute_date_range(timeline, start_date, end_date)
    start_date_obj = start_dt.date()
    end_date_obj = end_dt.date()

    # Determine clinician assigned patient IDs if doctor
    assigned_pt_ids = []
    if current_user.role == Role.DOCTOR.value:
        direct = db.scalars(select(Patient.id).where(Patient.assigned_doctor_id == current_user.id)).all()
        multi = db.scalars(
            select(PatientDoctorAssignment.patient_id).where(
                PatientDoctorAssignment.doctor_id == current_user.id,
                PatientDoctorAssignment.is_active == True
            )
        ).all()
        assigned_pt_ids = list(set(direct + multi))
        if patient_id and patient_id not in assigned_pt_ids:
            raise HTTPException(status_code=403, detail="Doctor can only access assigned patient analytics.")

    elif current_user.role == Role.PATIENT.value:
        pat = db.scalar(select(Patient).where(Patient.user_id == current_user.id))
        if not pat:
            raise HTTPException(status_code=404, detail="Patient record not found.")
        assigned_pt_ids = [pat.id]
        patient_id = pat.id

    # 1. Appointments Query
    appt_query = select(Appointment).where(
        Appointment.start_time >= start_dt,
        Appointment.start_time <= end_dt
    )
    if patient_id:
        appt_query = appt_query.where(Appointment.patient_id == patient_id)
    elif current_user.role == Role.DOCTOR.value:
        if assigned_pt_ids:
            appt_query = appt_query.where(
                or_(
                    Appointment.doctor_id == current_user.id,
                    Appointment.patient_id.in_(assigned_pt_ids)
                )
            )
        else:
            appt_query = appt_query.where(Appointment.doctor_id == current_user.id)
    elif current_user.role == Role.PATIENT.value and patient_id:
        appt_query = appt_query.where(Appointment.patient_id == patient_id)

    appts = db.scalars(appt_query.order_by(Appointment.start_time.asc())).all()

    completed_appts = sum(1 for a in appts if a.status == "completed")
    window_scheduled_appts = sum(1 for a in appts if a.status in ("scheduled", "rescheduled"))
    missed_appts = sum(1 for a in appts if a.status == "missed")
    reschedule_appts = sum(1 for a in appts if a.decision_status == "reschedule_requested")

    # Real-time upcoming appointments query (scheduled/rescheduled starting now or in the future)
    now_utc = dt.datetime.utcnow()
    upcoming_query = select(Appointment).where(
        Appointment.status.in_(["scheduled", "rescheduled"]),
        Appointment.start_time >= (now_utc - dt.timedelta(minutes=30))
    )
    if patient_id:
        upcoming_query = upcoming_query.where(Appointment.patient_id == patient_id)
    elif current_user.role == Role.DOCTOR.value:
        if assigned_pt_ids:
            upcoming_query = upcoming_query.where(
                or_(
                    Appointment.doctor_id == current_user.id,
                    Appointment.patient_id.in_(assigned_pt_ids)
                )
            )
        else:
            upcoming_query = upcoming_query.where(Appointment.doctor_id == current_user.id)
    elif current_user.role == Role.PATIENT.value and patient_id:
        upcoming_query = upcoming_query.where(Appointment.patient_id == patient_id)

    upcoming_appts_list = db.scalars(upcoming_query.order_by(Appointment.start_time.asc())).all()
    total_upcoming_count = len(upcoming_appts_list)

    # For 'today' view or when scheduled appointments are booked ahead in the calendar,
    # ensure upcoming scheduled appointments are accurately surfaced across all charts and KPIs.
    scheduled_appts = total_upcoming_count if timeline == "today" else max(window_scheduled_appts, total_upcoming_count)

    # 2. Clinical Documents Query
    doc_query = select(ClinicalDocument).where(
        ClinicalDocument.created_at >= start_dt,
        ClinicalDocument.created_at <= end_dt
    )
    if patient_id:
        if current_user.role == Role.PATIENT.value:
            doc_query = doc_query.where(ClinicalDocument.patient_id == patient_id, ClinicalDocument.status == DocStatus.APPROVED.value)
        else:
            doc_query = doc_query.where(ClinicalDocument.patient_id == patient_id)
    elif current_user.role == Role.DOCTOR.value:
        doc_filters = [
            ClinicalDocument.assigned_doctor_id == current_user.id,
            ClinicalDocument.uploaded_by == current_user.id
        ]
        if assigned_pt_ids:
            doc_filters.append(ClinicalDocument.patient_id.in_(assigned_pt_ids))
        doc_query = doc_query.where(or_(*doc_filters))
    elif current_user.role == Role.PATIENT.value and patient_id:
        doc_query = doc_query.where(ClinicalDocument.patient_id == patient_id, ClinicalDocument.status == DocStatus.APPROVED.value)

    docs = db.scalars(doc_query).all()
    approved_docs = sum(1 for d in docs if d.status == DocStatus.APPROVED.value)
    under_review_docs = sum(1 for d in docs if d.status in (DocStatus.UNDER_REVIEW.value, DocStatus.NEEDS_REVIEW.value))
    draft_docs = sum(1 for d in docs if d.status == DocStatus.DRAFT.value)
    rejected_docs = sum(1 for d in docs if d.status == DocStatus.REJECTED.value)

    # 3. Medication Adherence & Occurrences
    if current_user.role == Role.DOCTOR.value and not assigned_pt_ids and not patient_id:
        occurrences = []
    else:
        occ_query = select(MedicationOccurrence).where(
            MedicationOccurrence.scheduled_at >= start_dt,
            MedicationOccurrence.scheduled_at <= end_dt
        )
        if patient_id:
            occ_query = occ_query.where(MedicationOccurrence.patient_id == patient_id)
        elif current_user.role == Role.DOCTOR.value and assigned_pt_ids:
            occ_query = occ_query.where(MedicationOccurrence.patient_id.in_(assigned_pt_ids))
        elif current_user.role == Role.PATIENT.value and patient_id:
            occ_query = occ_query.where(MedicationOccurrence.patient_id == patient_id)

        occurrences = db.scalars(occ_query).all()

    total_doses_scheduled = len(occurrences)
    total_doses_taken = sum(1 for o in occurrences if o.state == OccurrenceState.TAKEN.value)
    total_doses_missed = sum(1 for o in occurrences if o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value))
    adherence_pct = round((total_doses_taken / total_doses_scheduled * 100), 1) if total_doses_scheduled > 0 else 0.0

    # 4. Daily Check-ins & Symptoms
    if current_user.role == Role.DOCTOR.value and not assigned_pt_ids and not patient_id:
        checkins = []
    else:
        checkin_query = select(DailyCheckIn).where(
            DailyCheckIn.checkin_date >= start_date_obj,
            DailyCheckIn.checkin_date <= end_date_obj
        )
        if patient_id:
            checkin_query = checkin_query.where(DailyCheckIn.patient_id == patient_id)
        elif current_user.role == Role.DOCTOR.value and assigned_pt_ids:
            checkin_query = checkin_query.where(DailyCheckIn.patient_id.in_(assigned_pt_ids))
        elif current_user.role == Role.PATIENT.value and patient_id:
            checkin_query = checkin_query.where(DailyCheckIn.patient_id == patient_id)

        checkins = db.scalars(checkin_query.order_by(DailyCheckIn.checkin_date.asc())).all()

    pain_scores = [c.pain_level for c in checkins if c.pain_level is not None]
    avg_pain = round(sum(pain_scores) / len(pain_scores), 1) if pain_scores else 0.0
    fever_count = sum(1 for c in checkins if c.fever)
    cough_count = sum(1 for c in checkins if c.cough)
    sleep_values = [c.sleep_hours for c in checkins if c.sleep_hours is not None]
    avg_sleep = round(sum(sleep_values) / len(sleep_values), 1) if sleep_values else 7.5

    # 5. Time-Series Construction (Real Bucketing)
    time_series = []
    if granularity == "hour":
        # 24 hour slots for single day
        for h in range(24):
            slot_start = start_dt.replace(hour=h, minute=0, second=0, microsecond=0)
            slot_end = start_dt.replace(hour=h, minute=59, second=59, microsecond=999999)
            h_label = f"{h:02d}:00"
            s_appts = sum(1 for a in appts if slot_start <= a.start_time <= slot_end)
            c_appts = sum(1 for a in appts if a.status == "completed" and slot_start <= a.start_time <= slot_end)
            t_meds = sum(1 for o in occurrences if o.state == OccurrenceState.TAKEN.value and slot_start <= o.scheduled_at <= slot_end)
            m_meds = sum(1 for o in occurrences if o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value) and slot_start <= o.scheduled_at <= slot_end)
            time_series.append({
                "label": h_label,
                "appointments_total": s_appts,
                "appointments_completed": c_appts,
                "medications_taken": t_meds,
                "medications_missed": m_meds,
                "activity_index": s_appts + t_meds + 1 if (s_appts + t_meds > 0) else 0
            })
    else:
        # Day-by-day slots
        num_days = max(1, (end_date_obj - start_date_obj).days + 1)
        for i in range(min(num_days, 31)):
            curr_date = start_date_obj + dt.timedelta(days=i)
            d_start = dt.datetime.combine(curr_date, dt.time.min)
            d_end = dt.datetime.combine(curr_date, dt.time.max)
            d_label = curr_date.strftime("%d %b")

            s_appts = sum(1 for a in appts if d_start <= a.start_time <= d_end)
            c_appts = sum(1 for a in appts if a.status == "completed" and d_start <= a.start_time <= d_end)
            t_meds = sum(1 for o in occurrences if o.state == OccurrenceState.TAKEN.value and d_start <= o.scheduled_at <= d_end)
            m_meds = sum(1 for o in occurrences if o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value) and d_start <= o.scheduled_at <= d_end)
            c_in = [c for c in checkins if c.checkin_date == curr_date]
            p_vals = [c.pain_level for c in c_in if c.pain_level is not None]
            p_level = round(sum(p_vals) / len(p_vals), 1) if p_vals else None

            time_series.append({
                "label": d_label,
                "date": curr_date.isoformat(),
                "appointments_total": s_appts,
                "appointments_completed": c_appts,
                "medications_taken": t_meds,
                "medications_missed": m_meds,
                "pain_level": p_level,
                "checkins_count": len(c_in)
            })

    # 6. Role-Specific Workflow Enrichment
    role_metrics: Dict[str, Any] = {}

    if current_user.role == Role.DOCTOR.value:
        active_pts_count = len(assigned_pt_ids)
        total_pending_prescriptions = 0
        if assigned_pt_ids:
            total_pending_prescriptions = db.scalar(
                select(func.count(ClinicalDocument.id)).where(
                    ClinicalDocument.status.in_([DocStatus.UNDER_REVIEW.value, DocStatus.NEEDS_REVIEW.value]),
                    ClinicalDocument.doc_type == "prescription",
                    or_(
                        ClinicalDocument.assigned_doctor_id == current_user.id,
                        ClinicalDocument.patient_id.in_(assigned_pt_ids)
                    )
                )
            ) or 0
        window_prescriptions = sum(1 for d in docs if d.doc_type == "prescription" and d.status in (DocStatus.UNDER_REVIEW.value, DocStatus.NEEDS_REVIEW.value))
        pending_prescriptions = max(window_prescriptions, total_pending_prescriptions)

        critical_risk_count = 0
        if assigned_pt_ids:
            latest_evals = db.execute(
                select(RiskClassificationHistory.patient_id, RiskClassificationHistory.level)
                .where(RiskClassificationHistory.patient_id.in_(assigned_pt_ids))
                .order_by(RiskClassificationHistory.patient_id, RiskClassificationHistory.evaluated_at.desc())
            ).all()
            seen_pts = set()
            for pid, lvl in latest_evals:
                if pid not in seen_pts:
                    seen_pts.add(pid)
                    if lvl == RiskLevel.RED.value or str(lvl).lower() == "red":
                        critical_risk_count += 1

        num_days = max(1, (end_date_obj - start_date_obj).days + 1)
        distinct_checkins = len({(c.patient_id, c.checkin_date) for c in checkins})
        expected_checkins = max(1, active_pts_count * num_days)
        doctor_compliance_pct = round(min(100.0, (distinct_checkins / expected_checkins) * 100.0), 1)

        role_metrics = {
            "caseload": active_pts_count,
            "assigned_patients_count": active_pts_count,
            "pending_prescription_reviews": pending_prescriptions,
            "consultation_sessions_count": len(appts),
            "critical_risk_patients": critical_risk_count,
            "daily_checkins_received": len(checkins),
            "compliance_percentage": doctor_compliance_pct,
        }

    elif current_user.role == Role.NURSE.value:
        total_pts = db.scalar(select(func.count(Patient.id)).where(Patient.active == True)) or 0
        assigned_subq = select(PatientDoctorAssignment.patient_id).where(PatientDoctorAssignment.is_active == True)
        unassigned_pts = db.scalar(
            select(func.count(Patient.id)).where(
                Patient.active == True,
                Patient.assigned_doctor_id.is_(None),
                Patient.id.notin_(assigned_subq)
            )
        ) or 0

        all_active_pids = db.scalars(select(Patient.id).where(Patient.active == True)).all()
        critical_risk_count = 0
        if all_active_pids:
            latest_evals = db.execute(
                select(RiskClassificationHistory.patient_id, RiskClassificationHistory.level)
                .where(RiskClassificationHistory.patient_id.in_(all_active_pids))
                .order_by(RiskClassificationHistory.patient_id, RiskClassificationHistory.evaluated_at.desc())
            ).all()
            seen_pts = set()
            for pid, lvl in latest_evals:
                if pid not in seen_pts:
                    seen_pts.add(pid)
                    if lvl == RiskLevel.RED.value or str(lvl).lower() == "red":
                        critical_risk_count += 1

        num_days = max(1, (end_date_obj - start_date_obj).days + 1)
        distinct_checkins = len({(c.patient_id, c.checkin_date) for c in checkins})
        expected_checkins = max(1, total_pts * num_days)
        compliance_pct = round(min(100.0, (distinct_checkins / expected_checkins) * 100.0), 1)

        role_metrics = {
            "caseload": total_pts,
            "active_patient_caseload": total_pts,
            "unassigned_patients_needing_triage": unassigned_pts,
            "critical_risk_patients": critical_risk_count,
            "daily_checkins_received": len(checkins),
            "compliance_percentage": compliance_pct
        }

    elif current_user.role == Role.PATIENT.value:
        role_metrics = {
            "caseload": 1,
            "adherence_percentage": adherence_pct,
            "doses_taken": total_doses_taken,
            "doses_scheduled": total_doses_scheduled,
            "average_pain_level": avg_pain,
            "average_sleep_hours": avg_sleep,
            "checkin_completed_in_window": len(checkins) > 0
        }

    elif current_user.role == Role.ADMIN.value:
        total_pts_admin = db.scalar(select(func.count(Patient.id)).where(Patient.active == True)) or 0
        total_docs_admin = db.scalar(select(func.count(User.id)).where(User.role == Role.DOCTOR.value, User.active == True)) or 0
        total_nurses_admin = db.scalar(select(func.count(User.id)).where(User.role == Role.NURSE.value, User.active == True)) or 0

        all_active_pids = db.scalars(select(Patient.id).where(Patient.active == True)).all()
        critical_risk_count = 0
        if all_active_pids:
            latest_evals = db.execute(
                select(RiskClassificationHistory.patient_id, RiskClassificationHistory.level)
                .where(RiskClassificationHistory.patient_id.in_(all_active_pids))
                .order_by(RiskClassificationHistory.patient_id, RiskClassificationHistory.evaluated_at.desc())
            ).all()
            seen_pts = set()
            for pid, lvl in latest_evals:
                if pid not in seen_pts:
                    seen_pts.add(pid)
                    if lvl == RiskLevel.RED.value or str(lvl).lower() == "red":
                        critical_risk_count += 1

        total_sms = db.scalar(
            select(func.count(SMSJob.id)).where(
                SMSJob.created_at >= start_dt,
                SMSJob.created_at <= end_dt
            )
        ) or 0
        delivered_sms = db.scalar(
            select(func.count(SMSJob.id)).where(
                SMSJob.created_at >= start_dt,
                SMSJob.created_at <= end_dt,
                SMSJob.state.in_([SMSJobStatus.SENT.value, SMSJobStatus.DELIVERED.value])
            )
        ) or 0
        failed_sms = db.scalar(
            select(func.count(SMSJob.id)).where(
                SMSJob.created_at >= start_dt,
                SMSJob.created_at <= end_dt,
                SMSJob.state == SMSJobStatus.FAILED.value
            )
        ) or 0
        audit_events = db.scalar(
            select(func.count(AuditLog.id)).where(
                AuditLog.created_at >= start_dt,
                AuditLog.created_at <= end_dt
            )
        ) or 0
        del_tickets_submitted = db.scalar(
            select(func.count(PatientDeletionTicket.id)).where(
                PatientDeletionTicket.created_at >= start_dt,
                PatientDeletionTicket.created_at <= end_dt
            )
        ) or 0
        role_metrics = {
            "caseload": total_pts_admin,
            "total_patients": total_pts_admin,
            "total_doctors": total_docs_admin,
            "total_nurses": total_nurses_admin,
            "critical_risk_patients": critical_risk_count,
            "sms_total_dispatched": total_sms,
            "sms_delivered_rate_pct": round((delivered_sms / max(1, total_sms)) * 100, 1) if total_sms > 0 else 100.0,
            "sms_failures": failed_sms,
            "audit_ledger_transactions": audit_events,
            "deletion_requests_in_window": del_tickets_submitted
        }

    # Detailed items for drilldown table with human-readable doctor and patient names
    target_appts = appts[:25] if appts else upcoming_appts_list[:25]
    doc_ids = {a.doctor_id for a in target_appts if a.doctor_id}
    pat_ids = {a.patient_id for a in target_appts if a.patient_id}

    doc_map = {}
    if doc_ids:
        docs_users = db.scalars(select(User).where(User.id.in_(list(doc_ids)))).all()
        doc_map = {u.id: u.name for u in docs_users}

    pat_map = {}
    if pat_ids:
        pts = db.scalars(select(Patient).where(Patient.id.in_(list(pat_ids)))).all()
        pt_user_ids = [p.user_id for p in pts if p.user_id]
        if pt_user_ids:
            pt_users = db.scalars(select(User).where(User.id.in_(pt_user_ids))).all()
            u_map = {u.id: u.name for u in pt_users}
            pat_map = {p.id: u_map.get(p.user_id, p.patient_code or "Patient") for p in pts}
        else:
            pat_map = {p.id: p.patient_code or "Patient" for p in pts}

    detailed_appointments = [
        {
            "id": a.id,
            "patient_id": a.patient_id,
            "patient_name": pat_map.get(a.patient_id, "Patient"),
            "doctor_id": a.doctor_id,
            "doctor_name": doc_map.get(a.doctor_id, "Specialist"),
            "start_time": a.start_time.isoformat(),
            "status": a.status,
            "decision_status": a.decision_status,
            "appointment_type": a.appointment_type or "general"
        }
        for a in target_appts
    ]

    return {
        "timeline": timeline,
        "start_time": start_dt.isoformat(),
        "end_time": end_dt.isoformat(),
        "window_label": window_label,
        "granularity": granularity,
        "role": current_user.role,
        "kpis": {
            "total_appointments": max(len(appts), completed_appts + missed_appts + scheduled_appts),
            "completed_appointments": completed_appts,
            "scheduled_appointments": scheduled_appts,
            "upcoming_appointments": total_upcoming_count,
            "missed_appointments": missed_appts,
            "reschedule_requests": reschedule_appts,
            "total_documents": len(docs),
            "approved_documents": approved_docs,
            "pending_documents": under_review_docs,
            "adherence_rate_pct": adherence_pct,
            "total_doses_scheduled": total_doses_scheduled,
            "total_doses_taken": total_doses_taken,
            "checkin_count": len(checkins),
            "average_pain_level": avg_pain,
            "fever_reports": fever_count,
            "cough_reports": cough_count,
            "average_sleep_hours": avg_sleep,
        },
        "role_metrics": role_metrics,
        "time_series": time_series,
        "charts": {
            "outcomes_breakdown": {
                "completed": completed_appts,
                "scheduled": scheduled_appts,
                "reschedule_requested": reschedule_appts,
                "missed": missed_appts
            },
            "documents_breakdown": {
                "approved": approved_docs,
                "under_review": under_review_docs,
                "draft": draft_docs,
                "rejected": rejected_docs
            },
            "adherence_rate_pct": adherence_pct
        },
        "recent_appointments": detailed_appointments
    }
