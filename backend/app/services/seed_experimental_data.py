"""
Recovera AI – Five-Patient Controlled Experimental Cohort Seeding Service
Seeds and manages the 5-patient research cohort with real uploaded/preserved documents,
OCR + NLP structured extractions, medication tracking, daily symptom check-ins,
digital twin state vectors, and explainable risk classifications.
"""

import os
import uuid
import logging
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.config import settings
from app.core.constants import Role, DocStatus, OccurrenceState, AppointmentStatus, RiskLevel
from app.models.user import User
from app.models.patient import Patient, PatientDoctorAssignment
from app.models.clinical_document import ClinicalDocument, OCRExtraction
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence
from app.models.checkin import DailyCheckIn
from app.models.appointment import Appointment
from app.models.digital_twin import RecoveryTrajectoryMilestone, DigitalTwinState
from app.ml.nlp_medical_extractor import NLPMedicalExtractor
from app.services.ocr_service import clean_ocr_text
from app.services.risk_service import evaluate_patient_risk
from app.services.digital_twin_service import compute_digital_twin_state, get_or_create_trajectory_milestones

logger = logging.getLogger("recovera.seed_experimental")

EXPERIMENTAL_PATIENTS_DEF = [
    {
        "code": "P-101",
        "name": "Rajesh Kumar",
        "email": "rajesh.kumar@patient.recovera.local",
        "age": 42,
        "gender": "Male",
        "condition": "Post-op Laparoscopic Appendectomy",
        "stage": "recovering",
        "care_status": "active",
        "history": {"allergies": "None", "surgery_date": "2026-11-08"},
        "doc_filename": "P-101_Appendectomy_Discharge_Rx.pdf",
        "raw_doc_text": "City Care Hospital - Department of Surgery\nPatient Name: Rajesh Kumar  Age: 42/M  ID: P-101\nDiagnosis: Post-op Laparoscopic Appendectomy\nRx:\nTab Paracetamol 650mg 1-0-1 after food for 5 days\nCap Amoxicillin 500mg 1-1-1 after food for 7 days\nTab Pan 40 40mg 1-0-0 before food for 5 days\nInstructions: Drink plenty of water and rest well.\nFollow-up: after 7 days",
        "med_profile": "high_adherence", # 95% adherence
        "checkin_profile": "steady_recovery", # pain 6 -> 4 -> 3 -> 2 -> 1, no fever
        "expected_risk": "green"
    },
    {
        "code": "P-102",
        "name": "Priya Sharma",
        "email": "priya.sharma@patient.recovera.local",
        "age": 58,
        "gender": "Female",
        "condition": "Type 2 Diabetes Mellitus with Post-Viral Asthenia",
        "stage": "active",
        "care_status": "active",
        "history": {"chronic": ["T2D", "Dyslipidemia"], "hba1c": "7.4%"},
        "doc_filename": "P-102_Endocrine_Prescription.pdf",
        "raw_doc_text": "Dr. Sarah Thomas, MD - Internal Medicine\nPatient: Priya Sharma  Age: 58 Y / Female  ID: P-102\nDiagnosis: Type 2 Diabetes Mellitus with Post-Viral Asthenia\nRx:\nTab Metformin 500mg 1-0-1 after food for 30 days\nTab Atorvastatin 20mg 0-0-1 night after food for 30 days\nTab Cetirizine 10mg 0-0-1 after food for 5 days\nReview in 1 month",
        "med_profile": "moderate_drop", # missed 2 evening doses, 72% adherence
        "checkin_profile": "moderate_fatigue", # pain 3/10, fatigue, no fever
        "expected_risk": "yellow"
    },
    {
        "code": "P-103",
        "name": "Madhavan S",
        "email": "madhavan@patient.recovera.local",
        "age": 34,
        "gender": "Male",
        "condition": "Dengue Fever",
        "stage": "recovering",
        "care_status": "active",
        "history": {"prior_fever": "102F on Nov 10", "platelets": "145k"},
        "doc_filename": "P-103_Infectious_Prescription.pdf",
        "raw_doc_text": "Apollo Telehealth Consultation Note\nPatient: Madhavan S  Age: 34 M  ID: P-103\nDiagnosis: Dengue Fever (Recovering)\nRx:\nTab Paracetamol 650mg 1-0-1 after food for 4 days\nCap Pantoprazole 40mg 1-0-0 before food for 5 days\nAdvice: Complete bed rest, oral rehydration solution, monitor temperature\nFollow-up: 2026-11-20",
        "med_profile": "single_day_lapse", # single day unlogged
        "checkin_profile": "recovering_fever_resolved", # pain 3/10, fever false
        "expected_risk": "yellow"
    },
    {
        "code": "P-104",
        "name": "Ananya Patel",
        "email": "ananya.patel@patient.recovera.local",
        "age": 49,
        "gender": "Female",
        "condition": "Post-Operative Cholecystectomy with Musculoskeletal Pain",
        "stage": "acute",
        "care_status": "active",
        "history": {"surgical_wound": "Mild redness observed"},
        "doc_filename": "P-104_Surgical_Followup_Rx.pdf",
        "raw_doc_text": "Department of Orthopedics & Surgery\nPatient Name: Ananya Patel  Age: 49/F  ID: P-104\nDiagnosis: Post-Operative Cholecystectomy with Musculoskeletal Pain\nRx:\nTab Deflazacort 6mg 1-0-0 after food for 5 days\nTab Tolperisone 150mg 1-0-1 after food for 5 days\nTab Pan 40 40mg 1-0-0 before food for 5 days\nTab Tramadol 50mg 0-0-1 after food for 3 days\nSos review if pain worsens",
        "med_profile": "severe_nonadherence", # 40% adherence, 3 missed doses
        "checkin_profile": "deterioration_fever", # pain 8/10, fever 101.8, restricted mobility
        "expected_risk": "red"
    },
    {
        "code": "P-105",
        "name": "Suresh Menon",
        "email": "suresh.menon@patient.recovera.local",
        "age": 62,
        "gender": "Male",
        "condition": "Essential Hypertension & Post-Knee Arthroplasty Recovery",
        "stage": "active",
        "care_status": "active",
        "history": {"hypertension": "10 years", "bp_target": "130/80"},
        "doc_filename": "P-105_Cardio_Ortho_Prescription.pdf",
        "raw_doc_text": "Heart Health Specialty Clinic\nPatient: Suresh Menon  Age: 62 Male  ID: P-105\nDiagnosis: Essential Hypertension & Post-Knee Arthroplasty Recovery\nRx:\nTab Losartan 50mg 1-0-0 after food for 30 days\nTab Amlodipine 5mg 0-0-1 night after food for 30 days\nTab Paracetamol 500mg 1-0-1 after food for 7 days\nReview after 2 weeks",
        "med_profile": "new_enrolled_awaiting", # awaiting start
        "checkin_profile": "no_checkins_yet", # newly enrolled
        "expected_risk": "insufficient_data"
    }
]


def seed_five_patient_experimental_cohort(db: Session, force_reset: bool = False) -> Dict[str, Any]:
    """
    Ensures all 5 experimental patients exist with full authentic clinical pipelines:
    Documents, OCR extractions, NLP entities, Medication occurrences, Check-ins, Digital Twin, and Risk records.
    """
    from passlib.context import CryptContext
    pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # Fetch attending doctor for assignment
    attending_doc = db.scalar(select(User).where(User.role == Role.DOCTOR.value).limit(1))
    if not attending_doc:
        attending_doc = User(
            id=str(uuid.uuid4()),
            email="dr.samuel@hospital.recovera.local",
            name="Dr. Samuel Roberts",
            role=Role.DOCTOR.value,
            specialization="General Medicine",
            password_hash=pwd_ctx.hash("Doctor@123"),
            active=True
        )
        db.add(attending_doc)
        db.commit()

    created_patients = []
    today = dt.date.today()
    now = dt.datetime.utcnow()

    for p_def in EXPERIMENTAL_PATIENTS_DEF:
        code = p_def["code"]
        existing_patient = db.scalar(select(Patient).where(Patient.patient_code == code))

        if existing_patient and not force_reset:
            created_patients.append(existing_patient)
            continue

        if existing_patient and force_reset:
            # Clean up prior records for clean reset
            db.query(MedicationOccurrence).filter(MedicationOccurrence.patient_id == existing_patient.id).delete()
            db.query(MedicationSchedule).filter(MedicationSchedule.patient_id == existing_patient.id).delete()
            db.query(Medication).filter(Medication.patient_id == existing_patient.id).delete()
            db.query(DailyCheckIn).filter(DailyCheckIn.patient_id == existing_patient.id).delete()
            db.query(Appointment).filter(Appointment.patient_id == existing_patient.id).delete()
            db.query(RecoveryTrajectoryMilestone).filter(RecoveryTrajectoryMilestone.patient_id == existing_patient.id).delete()
            db.query(DigitalTwinState).filter(DigitalTwinState.patient_id == existing_patient.id).delete()
            db.query(ClinicalDocument).filter(ClinicalDocument.patient_id == existing_patient.id).delete()
            db.query(PatientDoctorAssignment).filter(PatientDoctorAssignment.patient_id == existing_patient.id).delete()
            db.query(Patient).filter(Patient.id == existing_patient.id).delete()
            db.commit()

        # 1. Create User & Patient
        user = User(
            id=str(uuid.uuid4()),
            email=p_def["email"],
            name=p_def["name"],
            role=Role.PATIENT.value,
            password_hash=pwd_ctx.hash("Patient@123"),
            active=True
        )
        db.add(user)
        db.flush()

        # Enrollment date: 7 days ago for mature tracking, today for newly enrolled (P-105)
        days_ago = 0 if code == "P-105" else 6
        patient_created_at = now - dt.timedelta(days=days_ago)

        patient = Patient(
            id=str(uuid.uuid4()),
            patient_code=code,
            user_id=user.id,
            gender=p_def["gender"],
            active=True,
            care_status=p_def["care_status"],
            active_condition=p_def["condition"],
            condition_stage=p_def["stage"],
            previous_history=bool(p_def["history"]),
            history=p_def["history"],
            assigned_doctor_id=attending_doc.id,
            created_at=patient_created_at,
            updated_at=patient_created_at
        )
        db.add(patient)
        db.flush()

        # Doctor assignment
        assign = PatientDoctorAssignment(
            id=str(uuid.uuid4()),
            patient_id=patient.id,
            doctor_id=attending_doc.id,
            assigned_by=attending_doc.id,
            is_active=True,
            assigned_at=patient_created_at
        )
        db.add(assign)

        # 2. Upload & Process Document with OCR + NLP Pipeline
        raw_text = p_def["raw_doc_text"]
        cleaned_text = clean_ocr_text(raw_text)
        nlp_entities = NLPMedicalExtractor.extract_structured_clinical_entities(cleaned_text)

        doc_dir = Path(settings.UPLOAD_DIR) / patient.id
        doc_dir.mkdir(parents=True, exist_ok=True)
        doc_file_path = doc_dir / p_def["doc_filename"]
        with open(doc_file_path, "w", encoding="utf-8") as f:
            f.write(raw_text)

        doc = ClinicalDocument(
            id=str(uuid.uuid4()),
            patient_id=patient.id,
            doc_type="prescription",
            filename=p_def["doc_filename"],
            stored_path=str(doc_file_path),
            status=DocStatus.APPROVED.value,
            version=1,
            extracted_text=cleaned_text,
            structured_data=nlp_entities,
            uploaded_by=attending_doc.id,
            assigned_doctor_id=attending_doc.id,
            created_at=patient_created_at,
            updated_at=patient_created_at
        )
        db.add(doc)
        db.flush()

        ocr_record = OCRExtraction(
            document_id=doc.id,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            extracted_fields=nlp_entities,
            confidence_scores={"overall": nlp_entities.get("extraction_confidence", 0.95)},
            created_at=patient_created_at
        )
        db.add(ocr_record)

        # 3. Create Prescribed Medications & Schedules
        med_start_d = patient_created_at.date()
        for med_data in nlp_entities.get("medications", []):
            med = Medication(
                id=str(uuid.uuid4()),
                patient_id=patient.id,
                name=med_data["name"],
                generic_name=med_data.get("generic_name", med_data["name"]),
                strength=med_data.get("strength", "500mg"),
                dose=med_data.get("dose", "1 tablet"),
                start_date=med_start_d,
                end_date=med_start_d + dt.timedelta(days=med_data.get("duration_days", 5)),
                instructions=f"Take {med_data.get('dose', '1 tablet')} ({med_data.get('food_relation', 'after_food').replace('_', ' ')})",
                status="active" if code != "P-105" else "active",
                source_document_id=doc.id,
                created_by=attending_doc.id,
                created_at=patient_created_at
            )
            db.add(med)
            db.flush()

            for sched_data in med_data.get("schedules", [{"label": "Morning", "time_hhmm": "08:00"}]):
                sched = MedicationSchedule(
                    id=str(uuid.uuid4()),
                    medication_id=med.id,
                    patient_id=patient.id,
                    label=sched_data["label"],
                    time_hhmm=sched_data["time_hhmm"],
                    dose=med.dose,
                    food_relation=med_data.get("food_relation", "after_food")
                )
                db.add(sched)
                db.flush()

                # Generate occurrences if not P-105
                if code != "P-105":
                    for day_offset in range(days_ago):
                        occ_date = med_start_d + dt.timedelta(days=day_offset)
                        hh, mm = map(int, sched_data["time_hhmm"].split(":"))
                        occ_time = dt.datetime.combine(occ_date, dt.time(hh, mm))

                        # Determine state based on med_profile
                        if p_def["med_profile"] == "high_adherence":
                            st = OccurrenceState.TAKEN.value
                        elif p_def["med_profile"] == "moderate_drop":
                            # Miss evening doses on recent 2 days
                            st = OccurrenceState.MISSED.value if (sched_data["label"].lower() == "night" and day_offset >= days_ago - 2) else OccurrenceState.TAKEN.value
                        elif p_def["med_profile"] == "single_day_lapse":
                            st = OccurrenceState.MISSED.value if day_offset == days_ago - 1 else OccurrenceState.TAKEN.value
                        elif p_def["med_profile"] == "severe_nonadherence":
                            # Missed majority of recent doses
                            st = OccurrenceState.MISSED.value if day_offset >= 2 else OccurrenceState.TAKEN.value
                        else:
                            st = OccurrenceState.PENDING.value

                        occ = MedicationOccurrence(
                            id=str(uuid.uuid4()),
                            medication_id=med.id,
                            schedule_id=sched.id,
                            patient_id=patient.id,
                            scheduled_at=occ_time,
                            state=st
                        )
                        db.add(occ)

        # 4. Generate Daily Check-in Reports
        if code != "P-105":
            for day_offset in range(days_ago):
                chk_date = med_start_d + dt.timedelta(days=day_offset)

                if p_def["checkin_profile"] == "steady_recovery":
                    pain_val = max(1, 6 - day_offset)
                    fever_val = False
                    sleep_val = 7.5 + (day_offset * 0.2)
                    act_val = "light" if day_offset <= 2 else "normal"
                    notes = "Feeling steadily better. Wound clean."
                elif p_def["checkin_profile"] == "moderate_fatigue":
                    pain_val = 3
                    fever_val = False
                    sleep_val = 6.0
                    act_val = "light"
                    notes = "Feeling fatigued in the afternoons. Blood sugars stable."
                elif p_def["checkin_profile"] == "recovering_fever_resolved":
                    # Single-day lapse: yesterday not logged
                    if day_offset == days_ago - 1:
                        continue # Yesterday missing!
                    pain_val = 3
                    fever_val = False
                    sleep_val = 7.0
                    act_val = "rest"
                    notes = "Fever resolved. Drinking coconut water and fluids."
                elif p_def["checkin_profile"] == "deterioration_fever":
                    pain_val = 8
                    fever_val = True
                    sleep_val = 3.5
                    act_val = "bed_rest"
                    notes = "Severe pain at surgical site. Shivering and elevated fever 101.8F."
                else:
                    pain_val = 3
                    fever_val = False
                    sleep_val = 7.0
                    act_val = "normal"
                    notes = ""

                chk = DailyCheckIn(
                    id=str(uuid.uuid4()),
                    patient_id=patient.id,
                    checkin_date=chk_date,
                    pain_level=pain_val,
                    pain_location="Surgical site" if "Appendectomy" in p_def["condition"] or "Cholecystectomy" in p_def["condition"] else "General",
                    fever=fever_val,
                    cough=False,
                    sleep_hours=sleep_val,
                    activity_level=act_val,
                    notes=notes,
                    submitted_at=dt.datetime.combine(chk_date, dt.time(19, 30))
                )
                db.add(chk)

        db.commit()

        # 5. Initialize Trajectory Milestones, Digital Twin State, and Risk Engine
        get_or_create_trajectory_milestones(db, patient.id)
        compute_digital_twin_state(db, patient.id)
        evaluate_patient_risk(db, patient.id, force=True)

        created_patients.append(patient)

    db.commit()
    logger.info(f"Successfully seeded/verified 5-patient experimental cohort.")
    return {"status": "success", "patients_seeded": len(created_patients)}
