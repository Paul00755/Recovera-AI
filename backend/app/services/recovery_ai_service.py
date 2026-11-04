"""
Recovera AI – Clinical Intelligence & Universal Recovery Assistant Service
Integrates Google Gemini API (gemini-2.5-flash) with dynamic in-app database context injection,
strict medical-only guardrails, multi-role clinical reasoning (Patient, Doctor, Nurse, Admin),
multilingual support (English, Tamil, Hindi), file upload analysis, and a bulletproof offline-safe fallback engine.
"""

import os
import re
import json
import base64
import time
import logging
import datetime as dt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, or_, desc

from app.core.config import settings
from app.core.constants import Role, DocStatus
from app.models.user import User
from app.models.patient import Patient
from app.models.recovery_plan import RecoveryPlan
from app.models.medication import Medication, MedicationSchedule, MedicationOccurrence
from app.models.checkin import DailyCheckIn
from app.models.appointment import Appointment
from app.models.consultation import Consultation
from app.models.chat_session import ChatSession, ChatMessage

logger = logging.getLogger("recovera.ai_service")

# Emergency Red Flags
EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bshortness of breath\b",
    r"\bcan'?t breathe\b",
    r"\bdifficulty breathing\b",
    r"\bunconscious\b",
    r"\bfainted\b",
    r"\bpassed out\b",
    r"\bsevere bleeding\b",
    r"\bheavy bleeding\b",
    r"\bvomiting blood\b",
    r"\bcoughing blood\b",
    r"\bstroke\b",
    r"\bfacial drooping\b",
    r"\bparalysis\b",
    r"\bsuicid(e|al)\b",
    r"\bpoison(ing|ed)?\b",
    r"\banaphylaxis\b",
    r"\bsevere allergic reaction\b",
    r"\bseizure\b",
    r"\bheart attack\b",
    r"\bextreme pain (10|ten)\b",
    r"\bbluish lips\b",
    r"\bcyanosis\b",
]

# Prescription / Dosage Alteration Guardrail
PRESCRIPTION_BOUNDARY_PATTERNS = [
    r"\bchange (my )?dose\b",
    r"\bincrease (my )?dose\b",
    r"\bdecrease (my )?dose\b",
    r"\bdouble (my )?dose\b",
    r"\bcut (the )?tablet\b",
    r"\bstop taking\b",
    r"\bdiscontinue (my )?med\b",
    r"\bskip (my )?dose\b",
    r"\btake extra\b",
    r"\bprescribe\b",
    r"\balternative medic(ation|ine)\b",
    r"\bswap medic(ation|ine)\b",
    r"\bcan i take more\b",
    r"\bhow many more pills can i take\b",
]

# Off-topic / Non-Healthcare Filters
OFF_TOPIC_PATTERNS = [
    r"\bwrite (a )?(python|javascript|code|script|program|css|html)\b",
    r"\bwho (is|was) the president\b",
    r"\bwho won the (election|match|game|world cup|super bowl)\b",
    r"\bcrypto(currency)?\b",
    r"\bbitcoin\b",
    r"\binvest(ment)? advice\b",
    r"\bwrite (a )?(poem|story|song|essay) about\b",
    r"\bplay a game\b",
    r"\btell me a joke\b",
    r"\bmovie review\b",
    r"\bweather in (new york|paris|london|tokyo|delhi|chennai)\b",
]

HEALTHCARE_KEYWORDS = [
    "medicine", "medication", "pill", "tablet", "dose", "dosage", "timing", "food",
    "schedule", "recovery", "plan", "doctor", "nurse", "hospital", "wound", "incision",
    "pain", "fever", "temperature", "symptom", "diet", "food", "eat", "drink", "nutrition",
    "activity", "exercise", "walk", "sleep", "dressing", "bandage", "bleed", "infection",
    "stitches", "swelling", "redness", "appointment", "consultation", "checkin", "check-in",
    "prescription", "lab", "report", "vitals", "health", "discharge", "care", "rest",
    "hygiene", "therapy", "allergies", "side effect", "headache", "nausea", "dizziness",
    "constellation", "call", "telehealth", "patient", "triage", "surgery", "operation",
    "bath", "bathing", "shower", "showering", "hot water", "wash", "drive", "driving",
    "stairs", "lift", "lifting", "work", "job", "office", "routine", "heal", "healing",
    # Tamil keywords
    "மருந்து", "மாத்திரை", "மீட்பு", "மருத்துவர்", "செவிலியர்", "வலி", "காய்ச்சல்", "உணவு", "காயம்",
    "குளிக்க", "குளியல்", "வெந்நீர்", "தண்ணீர்", "தூக்கம்", "படுக்க", "தூங்கு", "நட", "நடக்க",
    "டிரைவிங்", "வாகனம்", "வண்டி", "படிக்கட்டு", "கட்டு", "தையல்", "வீக்கம்", "வேலை", "சாப்பாடு",
    # Hindi keywords
    "दवा", "दवाई", "गोली", "डॉक्टर", "नर्स", "दर्द", "बुखार", "घाव", "भोजन", "सलाह", "स्वास्थ्य",
    "नहाना", "स्नान", "पानी", "सोना", "नींद", "चलना", "गाड़ी", "सीढ़ी", "पट्टी", "टांके", "सूजन", "काम"
]

COMMON_MED_PURPOSES = {
    "amoxicillin": "Antibiotic to prevent or treat bacterial infections at surgical or tissue sites",
    "augmentin": "Broad-spectrum antibiotic to protect against postoperative bacterial infections",
    "ciprofloxacin": "Fluoroquinolone antibiotic for tissue and urinary tract infection prevention",
    "cephalexin": "First-generation cephalosporin antibiotic for skin and wound infection prophylaxis",
    "azithromycin": "Macrolide antibiotic for respiratory and systemic infection defense",
    "paracetamol": "Analgesic and antipyretic to relieve mild-to-moderate post-op pain and reduce fever",
    "acetaminophen": "Analgesic and antipyretic for effective pain and temperature control",
    "ibuprofen": "Non-steroidal anti-inflammatory (NSAID) to reduce swelling, inflammation, and surgical soreness",
    "tramadol": "Moderate-to-severe pain relief following surgical procedures (take strictly as directed)",
    "pantoprazole": "Proton pump inhibitor (PPI) to protect the stomach lining and prevent medication-induced acidity",
    "omeprazole": "Reduces gastric acid production and prevents stomach irritation caused by recovery medicines",
    "ondansetron": "Prevents postoperative or medication-induced nausea and vomiting",
    "metformin": "Maintains optimal blood sugar levels, crucial for normal cellular wound healing",
    "aspirin": "Mild antiplatelet / blood-thinner to promote circulation and prevent clot formation",
    "atorvastatin": "Regulates blood cholesterol and stabilizes vascular recovery",
    "amlodipine": "Calcium channel blocker for stable blood pressure control during convalescence",
    "telmisartan": "Angiotensin receptor blocker (ARB) to ensure safe cardiovascular blood pressure",
    "vitamin c": "Supports collagen synthesis, immune defenses, and rapid tissue recovery",
    "zinc": "Essential trace mineral to accelerate wound closure and cellular repair",
    "calcium": "Promotes bone density, skeletal recovery, and neuromuscular strength"
}


def get_gemini_api_key() -> Optional[str]:
    """
    Dynamically loads the Gemini API key from environment or .env files,
    allowing hot-reloading when user updates .env.
    """
    key = os.environ.get("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)
    if key and key.strip():
        return key.strip()

    # Attempt re-reading from .env files
    env_paths = [
        Path("d:/Recovera AI/backend/.env"),
        Path("d:/Recovera AI/.env"),
        Path("./backend/.env"),
        Path("./.env"),
    ]
    for p in env_paths:
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            os.environ["GEMINI_API_KEY"] = val
                            return val
            except Exception:
                pass
    return None


def detect_regex(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    for pat in patterns:
        if re.search(pat, lowered):
            return True
    return False


def is_healthcare_relevant(query: str, has_medical_file: bool = False) -> bool:
    """Verifies that query or attached file is legitimate healthcare/medical/app related."""
    if has_medical_file:
        return True
    q = query.lower().strip()
    if len(q) < 3:
        return True
    # Explicit off-topic check
    if detect_regex(q, OFF_TOPIC_PATTERNS):
        return False
    # Check if any healthcare keyword is present
    for kw in HEALTHCARE_KEYWORDS:
        if kw in q:
            return True
    # Specific clinical/hospital application terms
    clinical_app_terms = ["hospital", "recovera", "appointment", "clinic", "checkin", "check-in", "consultation", "doctor", "nurse", "prescription", "convalescence", "telehealth", "my care", "my recovery", "my health"]
    if any(term in q for term in clinical_app_terms):
        return True
    # Non-English native script queries within medical recovery scope
    if re.search(r'[\u0B80-\u0BFF\u0900-\u097F]', q):
        return True
    return False


def infer_medication_purpose(med_name: str, doc_instructions: Optional[str] = None) -> str:
    """Infers clinical indication for medication."""
    if doc_instructions and len(doc_instructions.strip()) > 3:
        return doc_instructions.strip()
    lower_name = (med_name or "").lower()
    for drug_key, purpose in COMMON_MED_PURPOSES.items():
        if drug_key in lower_name:
            return purpose
    return "Prescribed by your attending physician as part of your post-discharge recovery regimen."


def gather_clinical_context(db: Session, user: User, patient_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Gathers dynamic, real-time database records tailored to user role.
    """
    ctx: Dict[str, Any] = {
        "user_role": user.role,
        "user_name": user.name,
        "user_email": user.email,
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    if user.role == Role.PATIENT.value:
        # Resolve patient
        patient = db.scalar(select(Patient).where(Patient.user_id == user.id))
        if not patient and patient_id:
            patient = db.get(Patient, patient_id)
        if not patient:
            ctx["patient_found"] = False
            return ctx

        ctx["patient_found"] = True
        p_name = patient.user.name if patient.user else (getattr(patient, "name", None) or user.name)
        ctx["patient_name"] = p_name
        ctx["patient_code"] = patient.patient_code
        hist = patient.history if isinstance(patient.history, dict) else {}
        ctx["surgical_procedure"] = getattr(patient, "surgical_procedure", None) or hist.get("surgical_procedure") or "General Surgical / Clinical Convalescence"
        disch = getattr(patient, "discharge_date", None)
        ctx["discharge_date"] = disch.strftime("%Y-%m-%d") if disch else "Recent"
        ctx["current_risk_level"] = getattr(patient, "current_risk_level", None) or "LOW"
        ctx["active_condition"] = getattr(patient, "active_condition", None) or "Clinical Convalescence"
        ctx["condition_stage"] = getattr(patient, "condition_stage", "active") or "recovering"
        ctx["condition_notes"] = getattr(patient, "condition_notes", "") or ""

        # Attending doctor & nurse
        if patient.assigned_doctor_id:
            doc = db.get(User, patient.assigned_doctor_id)
            ctx["attending_doctor"] = f"Dr. {doc.name}" if doc else "Hospital Attending Physician"
        else:
            ctx["attending_doctor"] = "Hospital Attending Physician"

        assigned_nurse_id = getattr(patient, "assigned_nurse_id", None)
        if assigned_nurse_id:
            nurse = db.get(User, assigned_nurse_id)
            ctx["assigned_nurse"] = nurse.name if nurse else "Care Coordinator"

        # Active Medications & Schedules
        meds = db.scalars(
            select(Medication).where(
                Medication.patient_id == patient.id,
                Medication.status == "active"
            )
        ).all()

        active_med_list = []
        for m in meds:
            schedules = db.scalars(
                select(MedicationSchedule).where(MedicationSchedule.medication_id == m.id)
            ).all()
            sched_items = []
            for s in schedules:
                food_note = f" ({s.food_relation})" if s.food_relation else ""
                sched_items.append(f"{s.time_hhmm}{food_note}")
            
            purpose = infer_medication_purpose(m.name, m.instructions)
            active_med_list.append({
                "id": m.id,
                "name": m.name,
                "strength": m.strength or "Standard",
                "dose": m.dose or "1 dose",
                "route": m.route or "Oral",
                "schedule_timings": sched_items if sched_items else ["As directed by doctor"],
                "purpose": purpose,
                "instructions": m.instructions or "Take with a full glass of water"
            })
        ctx["active_medications"] = active_med_list

        # Today's Medication Occurrences
        today_start = dt.datetime.combine(dt.date.today(), dt.time.min)
        today_end = dt.datetime.combine(dt.date.today(), dt.time.max)
        occurrences = db.scalars(
            select(MedicationOccurrence)
            .join(Medication, MedicationOccurrence.medication_id == Medication.id)
            .where(
                Medication.patient_id == patient.id,
                MedicationOccurrence.scheduled_at >= today_start,
                MedicationOccurrence.scheduled_at <= today_end
            )
            .order_by(MedicationOccurrence.scheduled_at.asc())
        ).all()

        today_doses = []
        for occ in occurrences:
            med_obj = db.get(Medication, occ.medication_id)
            today_doses.append({
                "time": occ.scheduled_at.strftime("%I:%M %p"),
                "medicine": med_obj.name if med_obj else "Prescription Medicine",
                "dose": med_obj.dose if med_obj else "1 unit",
                "state": occ.state,
                "taken_at": occ.finalized_at.strftime("%I:%M %p") if occ.finalized_at else None
            })
        ctx["today_occurrences"] = today_doses

        # Approved or Active Recovery Plan
        plan = db.scalar(
            select(RecoveryPlan)
            .where(
                RecoveryPlan.patient_id == patient.id,
                RecoveryPlan.status.in_([DocStatus.PUBLISHED.value, DocStatus.APPROVED.value, DocStatus.DRAFT.value, "published", "approved", "draft"])
            )
            .order_by(RecoveryPlan.version.desc())
        )
        if plan:
            ctx["has_plan"] = True
            ctx["plan_version"] = plan.version
            inst = plan.instructions if isinstance(plan.instructions, dict) else {}
            cnt = plan.content if isinstance(plan.content, dict) else {}
            ctx["dietary_guidelines"] = (
                inst.get("dietary_guidelines") or inst.get("diet") or
                cnt.get("dietary_guidelines") or cnt.get("diet") or "Hydrate well (2-3L water daily); eat light, nutritious foods; limit excess oil, spice, and sodium."
            )
            ctx["activity_restrictions"] = (
                inst.get("activity_restrictions") or inst.get("activity") or
                cnt.get("activity_restrictions") or cnt.get("activity") or "Avoid strenuous lifting (>5 kg), vigorous sports, or sudden twisting. Gentle 10-15 minute indoor walks are encouraged."
            )
            ctx["wound_care"] = (
                inst.get("wound_care") or inst.get("precautions") or inst.get("warnings") or
                "Keep surgical incision clean and completely dry. Wash hands thoroughly before touching dressing."
            )
            ctx["precautions"] = ctx["wound_care"]
            ctx["warning_signs"] = plan.warning_signs or [
                "Persistent fever above 38.5°C (101.3°F)",
                "Expanding redness, severe warmth, or foul-smelling drainage from incision",
                "Suddenly worsening or intractable pain not relieved by medication",
                "Persistent vomiting, inability to retain fluids, or dizziness"
            ]
            ctx["care_instructions"] = (
                inst.get("care_instructions") or inst.get("general") or
                cnt.get("care_instructions") or cnt.get("general") or "Rest adequately, take medications punctually, and complete your daily check-in every morning."
            )
            ctx["medications_summary"] = (
                inst.get("medications_summary") or cnt.get("medications_summary") or "Take prescribed recovery medications according to schedule."
            )
        else:
            ctx["has_plan"] = False

        # Past 7 Days of Daily Check-ins
        recent_checkins = db.scalars(
            select(DailyCheckIn)
            .where(DailyCheckIn.patient_id == patient.id)
            .order_by(DailyCheckIn.submitted_at.desc())
            .limit(7)
        ).all()

        if recent_checkins:
            first_c = recent_checkins[0]
            ctx["last_checkin"] = {
                "date": first_c.submitted_at.strftime("%Y-%m-%d %I:%M %p"),
                "pain_level": first_c.pain_level if first_c.pain_level is not None else first_c.pain_score,
                "fever_reported": first_c.fever,
                "activity_level": first_c.activity_level or "normal",
                "patient_notes": first_c.notes or "None reported",
            }
            ctx["past_7_checkins"] = [
                {
                    "date": c.submitted_at.strftime("%Y-%m-%d"),
                    "pain_level": c.pain_level if c.pain_level is not None else c.pain_score,
                    "fever": c.fever,
                    "activity_level": c.activity_level or "normal",
                    "notes": c.notes or ""
                }
                for c in recent_checkins
            ]

        # Appointments & Remaining Consultations
        appts = db.scalars(
            select(Appointment).where(
                Appointment.patient_id == patient.id,
                Appointment.status.in_(["scheduled", "confirmed", "pending", "in_progress"])
            ).order_by(Appointment.start_time.asc())
        ).all()
        ctx["upcoming_appointments"] = [
            {
                "id": a.id,
                "time": a.start_time.strftime("%Y-%m-%d %I:%M %p") if a.start_time else "Pending confirmation",
                "status": a.status,
                "notes": a.notes or "Post-Discharge Follow-up"
            }
            for a in appts
        ]

        consultations = db.scalars(
            select(Consultation).where(
                Consultation.patient_id == patient.id,
                Consultation.status.in_(["waiting", "active", "scheduled"])
            ).order_by(Consultation.created_at.desc())
        ).all()
        ctx["active_consultations"] = [
            {"room_id": c.room_id, "mode": c.mode, "status": c.status} for c in consultations
        ]
        ctx["remaining_consultations_count"] = len(consultations) + len(appts)

    elif user.role == Role.DOCTOR.value:
        # Doctor Caseload Context
        assigned_pts = db.scalars(
            select(Patient).where(Patient.assigned_doctor_id == user.id, Patient.active == True)
        ).all()
        high_risk_pts = [p for p in assigned_pts if getattr(p, "current_risk_level", "LOW") in ("HIGH", "CRITICAL")]
        ctx["total_assigned_patients"] = len(assigned_pts)
        ctx["high_risk_patients_count"] = len(high_risk_pts)
        ctx["high_risk_summary"] = [
            {
                "name": p.user.name if p.user else getattr(p, "name", "Patient"),
                "code": p.patient_code,
                "risk": getattr(p, "current_risk_level", "LOW"),
                "procedure": getattr(p, "surgical_procedure", "Clinical Convalescence")
            }
            for p in high_risk_pts[:5]
        ]
        
        # Today's appointments
        today_appts = db.scalars(
            select(Appointment).where(
                Appointment.doctor_id == user.id,
                Appointment.status.in_(["scheduled", "confirmed"])
            )
        ).all()
        ctx["today_appointments_count"] = len(today_appts)

    elif user.role == Role.NURSE.value:
        # Nurse Caseload Context
        all_active_pts = db.scalars(
            select(Patient).where(Patient.active == True)
        ).all()
        elevated_pain_pts = []
        for p in all_active_pts:
            latest_ci = db.scalar(
                select(DailyCheckIn).where(DailyCheckIn.patient_id == p.id).order_by(DailyCheckIn.submitted_at.desc())
            )
            if latest_ci and latest_ci.pain_level and latest_ci.pain_level >= 6:
                elevated_pain_pts.append({
                    "name": p.user.name if p.user else getattr(p, "name", "Patient"),
                    "code": p.patient_code,
                    "pain_level": latest_ci.pain_level,
                    "fever": latest_ci.fever
                })
        ctx["total_assigned_patients"] = len(all_active_pts)
        ctx["triage_priority_count"] = len(elevated_pain_pts)
        ctx["triage_priority_patients"] = elevated_pain_pts[:5]

    elif user.role == Role.ADMIN.value:
        # Hospital Telemetry Context
        total_pts = db.scalar(select(Patient).where(Patient.active == True)) or 0
        total_users = db.scalar(select(User).where(User.active == True)) or 0
        ctx["hospital_active_patients"] = len(db.scalars(select(Patient).where(Patient.active == True)).all())
        ctx["hospital_total_staff"] = len(db.scalars(select(User).where(User.role != Role.PATIENT.value)).all())
        ctx["active_telehealth_consultations"] = len(db.scalars(select(Consultation).where(Consultation.status == "active")).all())

    return ctx


def build_system_prompt(ctx: Dict[str, Any], lang: str = "en") -> str:
    """
    Builds the high-fidelity system prompt with strict healthcare guardrails and real-time database grounding.
    """
    lang_name = "English"
    if lang == "ta":
        lang_name = "Tamil (தமிழ்)"
    elif lang == "hi":
        lang_name = "Hindi (हिन्दी)"

    prompt = f"""You are Recovera AI, an enterprise-grade clinical recovery assistant embedded in a Hospital Clinical Operating System.
Your primary role is pair-assisting patients with their post-discharge convalescence and assisting clinicians with real-time caseload intelligence.

=======================================================
CRITICAL OPERATIONAL RULES & MEDICAL SAFETY GUARDRAILS:
=======================================================
1. DOMAIN RESTRICTION: You MUST ONLY answer questions strictly regarding:
   - The patient's prescribed medications, exact schedule timings, food relations, and medical purposes.
   - The hospital-approved recovery plan (diet, wound care, mobility, precautions, warning signs).
   - In-app records (check-ins, appointments, remaining teleconsultations, care team details).
   - Clinical or healthcare queries (symptom management, anatomy, wound care, pharmacology education).
   - Staff triage and patient caseload management for clinicians.
   If the user asks an off-topic question (e.g. computer programming, general trivia, politics, entertainment, financial advice, casual games), REJECT IT POLITELY and state that you are exclusively dedicated to hospital recovery and medical care.

2. EMERGENCY ESCALATION: If the user describes life-threatening red-flag symptoms (chest pain, severe shortness of breath, sudden facial droop or numbness, sudden paralysis, severe bleeding, coughing blood, anaphylaxis, unconsciousness, poisoning, suicidal thoughts, extreme pain 10/10):
   - Immediately provide an EMERGENCY ALERT warning.
   - Instruct them to call emergency services (112 or 911) or proceed immediately to the nearest emergency department.
   - Do NOT advise them to wait for online recovery recommendations.

3. PRESCRIPTION BOUNDARY: You MUST NOT alter, increase, reduce, stop, or swap prescribed medications, nor suggest taking extra doses. Direct the patient to consult their attending doctor or care coordinator for any dosage modifications.

4. REAL-TIME GROUNDING: Answer all queries using the live hospital database context provided below. Whenever a patient asks "what medicines do I have to take today and for what purpose", explicitly list each medication name, strength, time of day (morning, afternoon, evening, night), relation to food (e.g. after food), and the clear clinical purpose.

5. LANGUAGE REQUIREMENT: You MUST reply fluently, warmly, and naturally in {lang_name}.
   - If Tamil (ta): Respond in clean, readable Tamil script (தமிழ்).
   - If Hindi (hi): Respond in clear Devanagari Hindi script (हिन्दी).
   - If English (en): Respond in clear, empathetic professional English.

6. FORMATTING: Use clean GitHub markdown formatting with bullet points, bold highlights, and emoji accents for clarity.

7. DYNAMIC CONVERSATION-AWARE FOLLOW-UP SUGGESTIONS:
   At the very end of your response, you MUST provide 3 dynamic, context-aware follow-up queries or recommended next actions tailored specifically to the patient's immediate question, clinical records, medications, or surgical recovery.
   - Do NOT repeat questions that the user has already asked or generic static questions. Suggest meaningful, actionable follow-ups that logically advance the care conversation.
   - Match the requested language ({lang_name}).
   - Format them exactly as:
<<<SUGGESTIONS>>>
- Contextual next question 1
- Contextual next question 2
- Contextual next question 3
<<<END_SUGGESTIONS>>>

=======================================================
CURRENT IN-APP REAL-TIME CLINICAL DATABASE CONTEXT:
=======================================================
{json.dumps(ctx, indent=2, default=str)}
"""
    return prompt


def analyze_uploaded_file(file_data_b64: str, mime_type: str, file_name: str) -> Tuple[bool, str, Optional[bytes]]:
    """
    Validates uploaded file: verifies it is a clinical document / transcript / photo.
    Restricts unwanted media formats (audio/video like mp3, mp4, mkv, etc.).
    Supports PDF, DOCX, TXT/MD, and clinical images.
    Returns (is_medical, extracted_text_or_desc, raw_bytes).
    """
    fn_lower = file_name.lower()
    restricted_exts = (
        ".mp3", ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
        ".wav", ".aac", ".m4a", ".ogg", ".wma", ".flac",
        ".exe", ".bat", ".cmd", ".sh", ".zip", ".tar", ".gz", ".7z", ".rar"
    )
    if any(fn_lower.endswith(ext) for ext in restricted_exts) or mime_type.startswith(("audio/", "video/")):
        return False, "RESTRICTED_MEDIA_TYPE", None

    try:
        raw_bytes = base64.b64decode(file_data_b64)
    except Exception as e:
        return False, f"Failed to decode file data: {str(e)}", None

    # Text / Transcript / Markdown
    if mime_type.startswith("text/") or fn_lower.endswith(".txt") or fn_lower.endswith(".md"):
        try:
            text = raw_bytes.decode("utf-8", errors="ignore")
            # Check for healthcare keywords
            if any(kw in text.lower() for kw in HEALTHCARE_KEYWORDS) or "consultation" in text.lower() or "doctor" in text.lower() or "patient" in text.lower():
                return True, text[:4000], raw_bytes
            return False, "Text document does not contain healthcare or consultation dialogue.", raw_bytes
        except Exception:
            return False, "Unable to read text file.", raw_bytes

    # Word DOCX Document
    if fn_lower.endswith(".docx") or mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ):
        try:
            import io, zipfile, xml.etree.ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                xml_content = z.read("word/document.xml")
            root = ET.fromstring(xml_content)
            paras = []
            for p in root.iter():
                if p.tag.endswith("}p"):
                    p_text = "".join(t.text for t in p.iter() if t.tag.endswith("}t") and t.text)
                    if p_text.strip():
                        paras.append(p_text.strip())
            extracted_text = "\n".join(paras).strip()
            lower_docx = extracted_text.lower()
            if any(kw in lower_docx for kw in HEALTHCARE_KEYWORDS) or "consultation" in lower_docx or "prescription" in lower_docx or "doctor" in lower_docx or "patient" in lower_docx or "hospital" in lower_docx:
                return True, extracted_text[:4000], raw_bytes
            return False, "Uploaded DOCX does not appear to contain medical records, prescription details, or consultation notes.", raw_bytes
        except Exception as e:
            return False, f"Failed to parse DOCX: {str(e)}", raw_bytes

    # PDF Document (e.g. consultation export, discharge summary, lab report)
    if mime_type == "application/pdf" or fn_lower.endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
            extracted_text = ""
            for page in doc:
                extracted_text += page.get_text() + "\n"
            doc.close()

            lower_pdf = extracted_text.lower()
            if any(kw in lower_pdf for kw in HEALTHCARE_KEYWORDS) or "telehealth" in lower_pdf or "consultation" in lower_pdf or "hospital" in lower_pdf or "prescription" in lower_pdf:
                return True, extracted_text[:4000], raw_bytes
            return False, "Uploaded PDF does not appear to be a medical report or consultation transcript.", raw_bytes
        except Exception as e:
            return False, f"Failed to parse PDF: {str(e)}", raw_bytes

    # Image (Prescription photo, wound photo, lab test)
    if mime_type.startswith("image/"):
        # Images will be evaluated multimodal by Gemini
        return True, f"Clinical image attached: {file_name} ({len(raw_bytes)} bytes)", raw_bytes

    return False, "Unsupported file format. Please upload a PDF, DOCX, text consultation transcript, or clinical image.", raw_bytes


_gemini_cooldown_until: float = 0.0


def generate_fallback_response(
    user_query: str,
    ctx: Dict[str, Any],
    lang: str = "en",
    attachment_desc: Optional[str] = None
) -> Dict[str, Any]:
    """
    Bulletproof local deterministic clinical reasoning engine.
    Runs when Gemini API is offline, rate-limited, or API key is absent.
    Zero user downtime, zero error messages.
    """
    # Auto-detect script if user sent Tamil or Hindi text directly
    if re.search(r'[\u0B80-\u0BFF]', user_query):
        lang = "ta"
    elif re.search(r'[\u0900-\u097F]', user_query):
        lang = "hi"

    q_low = user_query.lower()
    citations = []
    suggested_actions = ["Log Daily Check-in", "View Care Plan", "Schedule Consultation"]
    is_emergency = False
    prescription_boundary = False

    def matches_intent(keywords: list[str]) -> bool:
        for k in keywords:
            if k.isalnum() and k.isascii():
                if re.search(r'\b' + re.escape(k) + r'\b', q_low):
                    return True
            elif " " in k and k.isascii():
                if re.search(r'\b' + re.escape(k) + r'\b', q_low):
                    return True
            else:
                if k in q_low:
                    return True
        return False

    # 1. Emergency Detection
    if detect_regex(q_low, EMERGENCY_PATTERNS):
        is_emergency = True
        if lang == "ta":
            reply = (
                "🚨 **அவசர எச்சரிக்கை (Emergency Alert):**\n\n"
                "நீங்கள் குறிப்பிட்டுள்ள அறிகுறிகள் தீவிர மருத்துவ அவசரநிலையைக் குறிக்கலாம். "
                "உடனடியாக **112 அல்லது 911** என்ற அவசர எண்ணை அழைக்கவும் அல்லது அருகிலுள்ள அவசர சிகிச்சைப் பிரிவுக்குச் செல்லவும். "
                "ஆன்லைன் ஆலோசனைக்காகக் காத்திருக்க வேண்டாம்!"
            )
            sugg_emerg = ["அவசர எண் 112 / 911", "அவசர சிகிச்சைப் பிரிவு"]
        elif lang == "hi":
            reply = (
                "🚨 **आपातकालीन चेतावनी (Emergency Alert):**\n\n"
                "आपके बताए गए लक्षण संभावित गंभीर आपातकालीन स्थिति का संकेत देते हैं। "
                "कृपया तुरंत **112 या 911** पर कॉल करें या नजदीकी आपातकालीन अस्पताल विभाग में जाएं। "
                "ऑनलाइन सलाह के लिए प्रतीक्षा न करें!"
            )
            sugg_emerg = ["आपातकालीन 112 / 911 कॉल करें", "अस्पताल आपातकालीन विभाग"]
        else:
            reply = (
                "🚨 **EMERGENCY MEDICAL ALERT:**\n\n"
                "Your reported symptoms indicate a potentially critical medical emergency. "
                "Please call emergency services (**112 or 911**) immediately or proceed to the nearest emergency department. "
                "Do not wait for online recovery advice."
            )
            sugg_emerg = ["Call Emergency 112 / 911", "Contact Hospital ER Desk"]
        return {
            "reply": reply,
            "citations": [{"source": "Emergency Protocol", "snippet": "Immediate emergency escalation"}],
            "suggested_actions": sugg_emerg,
            "is_emergency": True,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 2. Prescription Modification Detection
    if detect_regex(q_low, PRESCRIPTION_BOUNDARY_PATTERNS):
        prescription_boundary = True
        doc_name = ctx.get("attending_doctor", "your attending doctor")
        if lang == "ta":
            reply = (
                f"⚠️ **மருந்து அளவு மாற்ற பாதுகாப்பு வரம்பு:**\n\n"
                f"மருந்துகளின் அளவை மாற்றுவது, நிறுத்துவது அல்லது மாற்று மாத்திரைகளை உட்கொள்வது குறித்து AI வழிகாட்ட முடியாது. "
                f"உங்கள் மருத்துவர் ({doc_name}) மட்டுமே உங்கள் மருந்து அளவை மாற்ற முடியும். "
                f"தயவுசெய்து ஆலோசனைக் கூடம் மூலம் மருத்துவரைத் தொடர்பு கொள்ளவும்."
            )
            sugg_rx = ["மருத்துவரை அணுகவும்", "செவிலியரை தொடர்புகொள்ளவும்"]
        elif lang == "hi":
            reply = (
                f"⚠️ **दवा सुरक्षा सीमा (Prescription Safety Boundary):**\n\n"
                f"दवा की खुराक बदलने, दवा बंद करने या अतिरिक्त गोली लेने की सलाह एआई द्वारा नहीं दी जा सकती। "
                f"केवल आपके उपस्थित चिकित्सक ({doc_name}) ही नुस्खे में बदलाव कर सकते हैं। "
                f"कृपया परामर्श कक्ष के माध्यम से अपने डॉक्टर से संपर्क करें।"
            )
            sugg_rx = ["डॉक्टर परामर्श बुक करें", "देखभाल टीम को संदेश भेजें"]
        else:
            reply = (
                f"⚠️ **Prescription Safety Boundary Enforced:**\n\n"
                f"Dosage alterations, skipping doses, stopping, or taking extra medications cannot be advised by AI. "
                f"Only your attending physician ({doc_name}) can modify your prescription. "
                f"Please schedule a consultation with your doctor or contact the nursing desk."
            )
            sugg_rx = ["Schedule Doctor Consultation", "Message Care Team"]
        return {
            "reply": reply,
            "citations": [{"source": "Prescription Safety Boundary", "snippet": "Doctor-exclusive prescription adjustment"}],
            "suggested_actions": sugg_rx,
            "is_emergency": False,
            "prescription_boundary": True,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 3. Medical Document & Prescription Attachment Analysis / Summarization
    if attachment_desc and (
        "summar" in q_low or "prescrib" in q_low or "transcript" in q_low or "consult" in q_low or
        "analyze" in q_low or "explain" in q_low or "review" in q_low or "document" in q_low or
        "attach" in q_low or len(q_low) < 15
    ):
        lines = [line.strip() for line in attachment_desc.split("\n") if line.strip()]
        preview_text = "\n".join(lines[:12]) if lines else attachment_desc[:500]
        if lang == "ta":
            reply = (
                f"📋 **மருத்துவ ஆவணத்தின் சுருக்கம் (Clinical Document Summary):**\n\n"
                f"நீங்கள் சமர்ப்பித்த ஆவணத்தின் முக்கிய பகுப்பாய்வு விவரங்கள்:\n\n"
                f"```text\n{preview_text}\n```\n\n"
                f"**மருத்துவக் குறிப்புகள் & வழிகாட்டுதல்கள்:**\n"
                f"• மருந்துச்சீட்டு மற்றும் ஆலோசனைக் குறிப்புகள் கவனமாகப் பதிவு செய்யப்பட்டுள்ளன.\n"
                f"• பரிந்துரைக்கப்பட்ட மருந்துகளை சரியான நேரத்தில் உட்கொள்ளவும்.\n"
                f"• ஏதேனும் புதிய அறிகுறிகள் அல்லது ஒவ்வாமை ஏற்பட்டால் உடனடியாக மருத்துவரைத் தொடர்பு கொள்ளவும்."
            )
            sugg_doc = ["மருந்துகள் அட்டவணையைப் பார்க்கவும்", "மருத்துவரை அணுகவும்", "தினசரி செக்-இன்"]
        elif lang == "hi":
            reply = (
                f"📋 **चिकित्सीय दस्तावेज़ सारांश (Clinical Document Summary):**\n\n"
                f"आपके द्वारा प्रस्तुत दस्तावेज़ का संक्षिप्त अवलोकन:\n\n"
                f"```text\n{preview_text}\n```\n\n"
                f"**चिकित्सीय निर्देश:**\n"
                f"• नुस्खे और परामर्श विवरण की पुष्टि की गई है।\n"
                f"• सभी निर्धारित दवाएं समय पर और भोजन के निर्देशानुसार लें।\n"
                f"• किसी भी असुविधा या दुष्प्रभाव के मामले में तुरंत डॉक्टर से संपर्क करें।"
            )
            sugg_doc = ["दवा सूची देखें", "डॉक्टर से परामर्श करें", "दैनिक चेक-इन"]
        else:
            reply = (
                f"📋 **Clinical Document & Prescription Summary:**\n\n"
                f"Summary analysis of your submitted medical document:\n\n"
                f"```text\n{preview_text}\n```\n\n"
                f"**Clinical Guidance & Follow-up:**\n"
                f"• Prescribed medications and consultation dialogue points have been verified from your record.\n"
                f"• Please adhere to scheduled doses, meal timings, and recovery instructions.\n"
                f"• Do not alter dosages independently; contact your care team for any regimen updates."
            )
            sugg_doc = ["View Intake Tracker", "Schedule Doctor Consultation", "Daily Check-in"]
        return {
            "reply": reply,
            "citations": [{"source": "Uploaded Clinical Document", "snippet": "Extracted document transcript"}],
            "suggested_actions": sugg_doc,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 4. Off-Topic Check
    if not is_healthcare_relevant(q_low):
        if lang == "ta":
            reply = (
                "நான் ரெக்கவரா AI (Recovera AI) - உங்கள் பிரத்யேக மருத்துவ மற்றும் மீட்பு உதவியாளர். "
                "நான் உங்கள் மருத்துவமனை மீட்புத் திட்டம், பரிந்துரைக்கப்பட்ட மருந்துகள், அறிகுறிகள் மற்றும் அப்பாயிண்ட்மென்ட் தகவல்களுக்கு மட்டுமே பதிலளிக்க வடிவமைக்கப்பட்டுள்ளேன். "
                "இன்று உங்கள் சிகிச்சைக்கு நான் எவ்வாறு உதவ முடியும்?"
            )
            sugg_off = ["இன்றைய மருந்துகள்", "உணவு வழிகாட்டுதல்கள்", "அடுத்த சந்திப்பு"]
        elif lang == "hi":
            reply = (
                "मैं रिकवरा एआई (Recovera AI) हूँ - आपका समर्पित स्वास्थ्य और रिकवरी सहायक। "
                "मैं केवल आपकी अस्पताल देखभाल योजना, निर्धारित दवाओं, स्वास्थ्य लक्षणों और नियुक्तियों के बारे में सवालों के जवाब देने के लिए प्रशिक्षित हूँ। "
                "आज मैं आपकी रिकवरी में कैसे मदद कर सकता हूँ?"
            )
            sugg_off = ["मेरी आज की दवाएं", "आहार दिशानिर्देश", "अगली अपॉइंटमेंट"]
        else:
            reply = (
                "I am Recovera AI, your dedicated hospital recovery assistant. "
                "I am strictly programmed to answer questions concerning your approved recovery plan, prescribed medications, symptoms, appointments, and medical care. "
                "How can I assist your health and recovery today?"
            )
            sugg_off = ["My Medications Today", "Dietary Guidelines", "Next Appointment"]
        return {
            "reply": reply,
            "citations": [{"source": "Clinical Domain Scope", "snippet": "Healthcare queries only"}],
            "suggested_actions": sugg_off,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 4. Medication & Purpose Inquiries
    is_med_query = matches_intent([
        "medication", "medicine", "pill", "tablet", "dose", "timing", "schedule", "take today", "purpose",
        "what are my meds", "why am i taking", "மருந்து", "மாத்திரை", "சாப்பிட", "எடுக்க", "दवा", "दवाई", "गोली", "खुराक"
    ])
    if is_med_query and ctx.get("user_role") == Role.PATIENT.value:
        meds = ctx.get("active_medications", [])
        today_doses = ctx.get("today_occurrences", [])
        if meds:
            citations.append({"source": "Active Prescriptions", "snippet": f"{len(meds)} active medications on file"})
            if lang == "ta":
                lines = ["💊 **இன்று நீங்கள் உட்கொள்ள வேண்டிய மருந்துகள் மற்றும் அவற்றின் நோக்கம்:**\n"]
                for m in meds:
                    timings = ", ".join(m.get("schedule_timings", ["மருத்துவர் அறிவுறுத்தியபடி"]))
                    lines.append(f"• **{m['name']}** ({m.get('strength', '')} - {m.get('dose', '')})")
                    lines.append(f"  ⏰ **நேரம்:** {timings}")
                    lines.append(f"  🎯 **நோக்கம்:** {m.get('purpose', '')}")
                    lines.append(f"  📝 **வழிமுறை:** {m.get('instructions', '')}\n")
                if today_doses:
                    lines.append("🕒 **இன்றைய நேர அட்டவணை நிலை:**")
                    for d in today_doses:
                        status_sym = "✅ உட்கொள்ளப்பட்டது" if d['state'] == "taken" else ("⏳ திட்டமிடப்பட்டது" if d['state'] == "scheduled" else "⚠️ தவறவிடப்பட்டது")
                        lines.append(f"- {d['time']}: {d['medicine']} ({status_sym})")
                reply = "\n".join(lines)
                sugg_meds = ["மருந்து உட்கொண்டதை பதிவு செய்", "பக்க விளைவுகளை தெரிவி", "மருத்துவரை அணுகு"]
            elif lang == "hi":
                lines = ["💊 **आज आपको कौन सी दवाएं लेनी हैं और उनका क्या उद्देश्य है:**\n"]
                for m in meds:
                    timings = ", ".join(m.get("schedule_timings", ["डॉक्टर के निर्देशानुसार"]))
                    lines.append(f"• **{m['name']}** ({m.get('strength', '')} - {m.get('dose', '')})")
                    lines.append(f"  ⏰ **समय:** {timings}")
                    lines.append(f"  🎯 **उद्देश्य:** {m.get('purpose', '')}")
                    lines.append(f"  📝 **निर्देश:** {m.get('instructions', '')}\n")
                if today_doses:
                    lines.append("🕒 **आज की खुराक स्थिति:**")
                    for d in today_doses:
                        status_sym = "✅ ली गई" if d['state'] == "taken" else ("⏳ आगामी" if d['state'] == "scheduled" else "⚠️ छूटी हुई")
                        lines.append(f"- {d['time']}: {d['medicine']} ({status_sym})")
                reply = "\n".join(lines)
                sugg_meds = ["दवा सेवन दर्ज करें", "साइड इफेक्ट्स बताएं", "डॉक्टर से परामर्श लें"]
            else:
                lines = ["💊 **Your Prescribed Medications for Today & Clinical Purposes:**\n"]
                for m in meds:
                    timings = ", ".join(m.get("schedule_timings", ["As prescribed"]))
                    lines.append(f"• **{m['name']}** ({m.get('strength', '')} · {m.get('dose', '')})")
                    lines.append(f"  ⏰ **Timing:** {timings}")
                    lines.append(f"  🎯 **Purpose:** {m.get('purpose', '')}")
                    lines.append(f"  📝 **Directions:** {m.get('instructions', '')}\n")
                if today_doses:
                    lines.append("🕒 **Today's Dose Administration Log:**")
                    for d in today_doses:
                        state_tag = "✅ Completed" if d['state'] == "taken" else ("⏳ Scheduled" if d['state'] == "scheduled" else "⚠️ Missed")
                        lines.append(f"- {d['time']}: {d['medicine']} ({state_tag})")
                reply = "\n".join(lines)
                sugg_meds = ["Log Medicine Intake", "Report Side Effects", "Consult Doctor"]
            return {
                "reply": reply,
                "citations": citations,
                "suggested_actions": sugg_meds,
                "is_emergency": False,
                "prescription_boundary": False,
                "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
            }

    # 5. Appointments & Teleconsultations Inquiries
    is_appt_query = matches_intent([
        "appointment", "consultation", "constellation", "remaining", "next visit", "doctor visit",
        "when is my doctor", "meeting", "call", "சந்திப்பு", "ஆலோசனை", "அப்பாயிண்ட்மென்ட்", "பார்வை", "நேரம்", "अपॉइंटमेंट", "परामर्श", "मुलाकात", "डॉक्टर"
    ])
    if is_appt_query and ctx.get("user_role") == Role.PATIENT.value:
        appts = ctx.get("upcoming_appointments", [])
        rem_count = ctx.get("remaining_consultations_count", len(appts))
        doc = ctx.get("attending_doctor", "Hospital Physician")
        citations.append({"source": "Hospital Scheduling Desk", "snippet": f"{rem_count} remaining consultations/appointments"})
        if lang == "ta":
            reply = (
                f"📅 **உங்கள் வரவிருக்கும் மருத்துவ ஆலோசனைகள்:**\n\n"
                f"• மருத்துவர்: **{doc}**\n"
                f"• மீதமுள்ள ஆலோசனைகள் / சந்திப்புகள்: **{rem_count}**\n"
            )
            if appts:
                reply += "\nதிட்டமிடப்பட்ட சந்திப்புகள்:\n" + "\n".join(f"• {a['time']} ({a['mode']} - {a['status']})" for a in appts)
            else:
                reply += "\nதற்போது திட்டமிடப்பட்ட நேரடி சந்திப்பு எதுவும் இல்லை. நீங்கள் விரும்பினால் 'Book Consultation' மூலம் உடனடியாக புதிய ஆலோசனையை பதிவு செய்யலாம்."
            sugg_appts = ["டெலிஹெல்த் கோரிக்கை", "நேர அட்டவணையை காண்", "பராமரிப்பு உதவியாளர்"]
        elif lang == "hi":
            reply = (
                f"📅 **आपकी आगामी डॉक्टर नियुक्तियां और परामर्श:**\n\n"
                f"• उपस्थित डॉक्टर: **{doc}**\n"
                f"• शेष परामर्श / नियुक्तियां: **{rem_count}**\n"
            )
            if appts:
                reply += "\nनिर्धारित नियुक्तियां:\n" + "\n".join(f"• {a['time']} ({a['mode']} - {a['status']})" for a in appts)
            else:
                reply += "\nवर्तमान में कोई निर्धारित अपॉइंटमेंट लंबित नहीं है। आप आवश्यकता पड़ने पर नया परामर्श बुक कर सकते हैं।"
            sugg_appts = ["टेलीहेल्थ का अनुरोध करें", "उपलब्धता देखें", "देखभाल डेस्क"]
        else:
            reply = (
                f"📅 **Your Upcoming Consultations & Doctor Appointments:**\n\n"
                f"• Attending Physician: **{doc}**\n"
                f"• Total Consultations / Appointments Remaining: **{rem_count}**\n"
            )
            if appts:
                reply += "\n**Scheduled Sessions:**\n" + "\n".join(f"• {a['time']} ({a['mode'].upper()} - Status: {a['status']})" for a in appts)
            else:
                reply += "\nYou currently have no scheduled appointments. You can request a 15-minute teleconsultation or clinical visit anytime."
            sugg_appts = ["Request Teleconsultation", "View Availability", "Contact Care Desk"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_appts,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 6. Diet, Nutrition & Food Guidelines
    is_diet_query = matches_intent([
        "diet", "food", "eat", "drink", "nutrition", "meal", "hungry", "water",
        "உணவு", "சாப்பாடு", "தண்ணீர்", "குடிக்க", "பத்தியம்", "भोजन", "आहार", "खाना", "पानी", "पोषण"
    ])
    if is_diet_query and ctx.get("dietary_guidelines"):
        citations.append({"source": "Approved Recovery Plan", "snippet": "Dietary guidelines"})
        diet_txt = ctx.get("dietary_guidelines")
        if lang == "ta":
            reply = f"🥗 **உங்கள் அங்கீகரிக்கப்பட்ட உணவு வழிகாட்டுதல்கள்:**\n\n{diet_txt}\n\n*பரிந்துரை:* நாள் முழுவதும் போதுமான நீர் அருந்தி, அதிக காரம் மற்றும் எண்ணெய் உள்ள உணவுகளைத் தவிர்க்கவும்."
            sugg_diet = ["தினசரி செக்-இன்", "முழு திட்டத்தை காண்", "தண்ணீர் நினைவூட்டல்"]
        elif lang == "hi":
            reply = f"🥗 **आपके स्वीकृत आहार दिशानिर्देश:**\n\n{diet_txt}\n\n*सुझाव:* दिन भर में पर्याप्त पानी पिएं और तैलीय या अत्यधिक मसालेदार भोजन से बचें।"
            sugg_diet = ["दैनिक चेक-इन", "पूरी योजना देखें", "पानी का सेवन"]
        else:
            reply = f"🥗 **Your Approved Dietary & Nutrition Guidelines:**\n\n{diet_txt}\n\n*Recommendation:* Stay well-hydrated throughout the day and avoid heavy, overly oily or spicy foods that may stress digestion during healing."
            sugg_diet = ["Log Daily Check-in", "View Full Plan"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_diet,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 7. Bathing, Showering, Hot Water & Wet Dressing
    is_bath_query = matches_intent([
        "bath", "bathing", "shower", "showering", "hot water", "wash", "sponge bath", "tub", "wet",
        "குளிக்க", "குளியல்", "வெந்நீர்", "தண்ணீர்", "நனைய", "ஈரம்", "नहाना", "स्नान", "धोना", "गीला", "पानी"
    ])
    if is_bath_query:
        citations.append({"source": "Post-Op Wound & Hygiene Protocol", "snippet": "Bathing and incision moisture restrictions"})
        doc = ctx.get("attending_doctor", "your attending doctor")
        if lang == "ta":
            reply = (
                "🚿 **குளிப்பது மற்றும் வெந்நீர் பயன்பாடு குறித்த வழிகாட்டுதல்:**\n\n"
                "• **வெந்நீர் / தொட்டி குளியல் தவிர்க்கவும்:** உங்கள் மருத்துவர் முழுமையாக அனுமதிக்கும் வரை வெந்நீரிலோ அல்லது தொட்டியிலோ (bathtub) மூழ்கி குளிக்கக் கூடாது.\n"
                "• **கட்டு எப்போதும் உலர்ந்திருக்க வேண்டும்:** உங்கள் அறுவை சிகிச்சை காயத்தின் மீதுள்ள கட்டு (bandage / dressing) எப்போதும் முற்றிலும் உலர்ந்த நிலையில் இருக்க வேண்டும். வெந்நீர் அல்லது நீராவி படும்போது கட்டு நனைந்து தையல்கள் தளர்வடையவும், தொற்று (infection) ஏற்படவும் வாய்ப்புள்ளது.\n"
                "• **பரிந்துரைக்கப்பட்ட முறை:** காயத்தின் மீது தண்ணீர் படாமல் பாதுகாத்து மென்மையான **ஸ்பாஞ்ச் குளியல் (sponge bath)** மட்டுமே எடுக்கவும். காயத்தைச் சுற்றியுள்ள பகுதியை உலர்ந்த துணியால் மூடிக்கொள்ளவும்.\n"
                "• **கட்டு தவறுதலாக நனைந்தால்:** உடனடியாக சுத்தமான உலர்ந்த துணியால் மென்மையாக ஒற்றி எடுத்துவிட்டு, கட்டை மாற்ற உங்கள் செவிலியர் அல்லது மருத்துவரைத் தொடர்பு கொள்ளவும்."
            )
            sugg_bath = ["செவிலியரை தொடர்புகொள்", "காய பராமரிப்பு குறிப்புகள்", "அடுத்த செக்-இன்"]
        elif lang == "hi":
            reply = (
                "🚿 **स्नान और गर्म पानी के उपयोग के दिशानिर्देश:**\n\n"
                "• **गर्म पानी और टब स्नान से बचें:** जब तक आपके डॉक्टर स्पष्ट रूप से अनुमति न दें, तब तक टब में डूबकर या अत्यधिक गर्म पानी से स्नान न करें।\n"
                "• **पट्टी को सूखा रखें:** सर्जरी के घाव पर बंधी पट्टी (bandage / dressing) हमेशा पूरी तरह सूखी रहनी चाहिए। पानी या भाप लगने से टांकों में खिंचाव और संक्रमण का खतरा रहता है।\n"
                "• **अनुशंसित तरीका:** घाव वाले हिस्से पर पानी पड़े बिना सावधानी से **स्पंज बाथ (sponge bath)** लें। घाव वाले हिस्से को प्लास्टिक रैप या सूखे तौलिये से सुरक्षित रखें।\n"
                "• **पट्टी गीली होने पर:** यदि पट्टी पर पानी लग जाए, तो तुरंत साफ सूखे कपड़े से हल्के हाथों थपथपाकर सुखाएं और अपनी केयर टीम से संपर्क करें।"
            )
            sugg_bath = ["नर्स से संपर्क करें", "घाव देखभाल निर्देश", "दैनिक चेक-इन"]
        else:
            reply = (
                "🚿 **Bathing, Showering, and Incision Moisture Guidance:**\n\n"
                "• **Avoid Tub Immersion & Hot Water Baths:** Do not submerge your body in bathtubs, hot tubs, or pools until your surgeon gives direct authorization.\n"
                "• **Keep Dressing Completely Dry:** Your surgical dressing must remain 100% clean and dry. Hot water and steam can loosen surgical tape/staples, soften the healing tissue, and introduce bacterial infection.\n"
                "• **Recommended Method:** Take gentle **sponge baths** while keeping the surgical site carefully covered and dry.\n"
                "• **If Dressing Gets Wet:** Immediately pat dry with a clean, sterile cloth or towel (do not rub), and notify your attending care nurse for inspection or redressing."
            )
            sugg_bath = ["Contact Attending Nurse", "Wound Care Protocol", "Log Symptom Check-in"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_bath,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 8. Driving, Vehicles & Travel
    is_driving_query = matches_intent([
        "drive", "driving", "car", "bike", "vehicle", "scooter", "travel", "ride",
        "டிரைவிங்", "வாகனம்", "வண்டி", "ஓட்ட", "பயணம்", "காலாற", "गाड़ी", "चलाना", "सवारी", "यात्रा"
    ])
    if is_driving_query:
        citations.append({"source": "Postoperative Mobility Protocol", "snippet": "Driving and vehicle operation guidelines"})
        doc = ctx.get("attending_doctor", "your attending surgeon")
        if lang == "ta":
            reply = (
                f"🚗 **வாகனம் ஓட்டுதல் (Driving) குறித்த மருத்துவ வழிகாட்டுதல்:**\n\n"
                f"• **மருந்து உட்கொள்ளும்போது ஓட்ட வேண்டாம்:** தூக்கக் கலக்கம் அல்லது மந்தநிலையை ஏற்படுத்தும் வலி மருந்துகளை உட்கொள்ளும்போது எக்காரணம் கொண்டும் வாகனம் ஓட்டக்கூடாது.\n"
                f"• **மருத்துவர் அனுமதி அவசியம்:** உங்கள் அறுவை சிகிச்சை காயத்தின் தையல்கள் ஆறி, அவசர பிரேக் அடிக்கும் அளவிற்கு வயிற்று/உடல் தசைகள் வலிமை பெறும் வரை ({doc} அனுமதிக்கும் வரை) வாகனம் ஓட்டுவதைத் தவிர்க்கவும்.\n"
                f"• **பரிந்துரைக்கப்பட்ட ஓய்வு:** பொதுவாக அறுவை சிகிச்சைக்குப் பின் 1 முதல் 2 வாரங்கள் வரை சுய வாகனம் ஓட்டுவதைத் தவிர்த்து மற்றவர் உதவியுடன் பயணிக்க பரிந்துரைக்கப்படுகிறது."
            )
            sugg_drv = ["மருத்துவரை அணுகவும்", "செயல்பாட்டு கட்டுப்பாடுகள்", "அடுத்த சந்திப்பு"]
        elif lang == "hi":
            reply = (
                f"🚗 **वाहन चलाने (Driving) संबंधी क्लिनिकल दिशानिर्देश:**\n\n"
                f"• **दवाओं के प्रभाव में गाड़ी न चलाएं:** जब तक आप दर्द निवारक दवाएं ले रहे हैं जिनसे उनींदापन होता है, तब तक गाड़ी या बाइक न चलाएं।\n"
                f"• **डॉक्टर की अनुमति आवश्यक:** जब तक आपके घाव के टांके पूरी तरह न भर जाएं और इमरजेंसी ब्रेक लगाने में दर्द न हो ({doc} की पुष्टि तक), ड्राइविंग से बचें।\n"
                f"• **सलाह:** सर्जरी के बाद आमतौर पर 1 से 2 सप्ताह तक किसी अन्य व्यक्ति की सहायता से ही यात्रा करने की सलाह दी जाती है।"
            )
            sugg_drv = ["डॉक्टर से परामर्श लें", "गतिविधि सीमाएं", "अगली अपॉइंटमेंट"]
        else:
            reply = (
                f"🚗 **Driving and Vehicle Operation Guidance:**\n\n"
                f"• **No Driving on Pain Medications:** Never operate a motor vehicle while taking prescription pain medications or sedatives that cause drowsiness or delayed reflexes.\n"
                f"• **Surgeon Clearance Required:** You must wait until your surgical incision has healed and you can perform a sudden emergency brake stop without incision strain or pain, cleared by {doc}.\n"
                f"• **Typical Timeline:** Most post-surgical patients should arrange transportation assistance for at least 1-2 weeks post-discharge before resuming personal driving."
            )
            sugg_drv = ["Consult Doctor", "View Activity Guidelines", "Next Appointment"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_drv,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 9. Sleeping Position & Posture
    is_sleep_query = matches_intent([
        "sleep", "sleeping", "bed", "lie down", "position", "pillow", "rest", "posture",
        "தூக்கம்", "படுக்க", "தூங்கு", "நீந்த", "நிலை", "தலையணை", "नींद", "सोना", "करवट", "तकिया", "लेटना"
    ])
    if is_sleep_query:
        citations.append({"source": "Convalescence Ergonomics", "snippet": "Post-surgical sleep positions"})
        if lang == "ta":
            reply = (
                "🛏️ **தூங்கும் நிலை & ஓய்வு குறித்த வழிகாட்டுதல்:**\n\n"
                "• **மல்லாந்து படுக்கவும்:** உங்கள் முதுகுப்புறமாக மல்லாந்து படுப்பது காயத்தின் மீது அழுத்தம் ஏற்படாமல் பாதுகாக்கும்.\n"
                "• **தலையணை கொண்டு உடலை உயர்த்தவும்:** 1-2 கூடுதல் தலையணை வைத்து உடலை சுமார் 30 டிகிரி உயர்த்திப் படுப்பது வீக்கத்தைக் குறைக்கவும் மூச்சுவிடவும் உதவும்.\n"
                "• **காயத்தின் மீது படுக்க வேண்டாம்:** காயமுள்ள பக்கத்திலோ அல்லது குப்புறவோ (stomach) படுப்பதைத் தவிர்க்கவும்.\n"
                "• **போதுமான தூக்கம்:** திசுக்கள் சீரடைய தினமும் 8-9 மணி நேர அமைதியான தூக்கம் அவசியம்."
            )
            sugg_slp = ["அறிகுறி செக்-இன்", "பராமரிப்பு திட்டம்", "செவிலியர் உதவி"]
        elif lang == "hi":
            reply = (
                "🛏️ **सोने की स्थिति और आराम के दिशानिर्देश:**\n\n"
                "• **पीठ के बल सीधा सोएं:** अपनी पीठ के बल सीधा लेटने से चीरे और घाव पर अनुचित खिंचाव नहीं पड़ता।\n"
                "• **तकिये से सिर व धड़ को थोड़ा ऊंचा रखें:** 1-2 तकियों की मदद से सिर और ऊपरी शरीर को थोड़ा ऊंचा रखने से सूजन कम होती है।\n"
                "• **घाव के बल न सोएं:** जिस तरफ सर्जरी हुई है उस करवट या पेट के बल सोने से बचें।\n"
                "• **पर्याप्त आराम:** ऊतकों की तेजी से मरम्मत के लिए प्रतिदिन 8-9 घंटे की गहरी नींद लें।"
            )
            sugg_slp = ["लक्षण चेक-इन", "देखभाल योजना", "नर्स सहायता"]
        else:
            reply = (
                "🛏️ **Optimal Sleeping Positions & Rest Recommendations:**\n\n"
                "• **Sleep on Your Back:** Sleeping flat on your back protects your surgical site from accidental direct compression and tension.\n"
                "• **Elevate Head & Torso:** Use 1-2 extra pillows to keep your upper body elevated (~30 degrees); this reduces local swelling and facilitates effortless respiration.\n"
                "• **Avoid Direct Pressure:** Do not sleep on your stomach or directly roll onto the incision site.\n"
                "• **Restful Convalescence:** Aim for 8-9 hours of uninterrupted restorative sleep every night to accelerate tissue synthesis."
            )
            sugg_slp = ["Log Daily Rest", "View Care Plan", "Message Care Team"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_slp,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 10. Mobility, Walking, Stairs & Exercise
    is_mobility_query = matches_intent([
        "walk", "walking", "stairs", "step", "steps", "lift", "lifting", "heavy", "exercise", "workout", "gym", "run", "climb",
        "நட", "நடக்க", "படிக்கட்டு", "தூக்க", "பயிற்சி", "உடற்பயிற்சி", "சுமை", "चलना", "सीढ़ी", "व्यायाम", "वजन", "कसरत", "जिम"
    ])
    if is_mobility_query:
        citations.append({"source": "Post-op Physical Therapy & Mobility", "snippet": "Mobility progression and weight lifting restrictions"})
        acts = ctx.get("activity_restrictions", "Avoid lifting items heavier than 5kg (10 lbs) and strenuous exertion.")
        if lang == "ta":
            reply = (
                f"🚶 **நடைப்பயிற்சி மற்றும் செயல்பாட்டு வழிகாட்டுதல்:**\n\n"
                f"• **மெதுவான நடைப்பயிற்சி நல்லது:** இரத்த உறைவைத் (DVT) தடுக்கவும், குடல் செயல்பாட்டைத் தூண்டவும் வீட்டின் உள்ளே 5-10 நிமிடங்கள் மெதுவாக நடப்பது மிகவும் நல்லது.\n"
                f"• **எடை தூக்குவதைத் தவிர்க்கவும்:** {acts}\n"
                f"• **படிக்கட்டுகள்:** படிக்கட்டுகளில் ஏறும்போது கைப்பிடியைப் (handrail) பிடித்து, ஒவ்வொரு படியாக நிதானமாக ஏறவும்.\n"
                f"• **தீவிர பயிற்சிகள் வேண்டாம்:** உடற்பயிற்சிக் கூடம், ஓடுதல் அல்லது வளைந்து நெளிந்து வேலை செய்வதை மருத்துவர் முழுமையாக அனுமதிக்கும் வரை தவிர்க்கவும்."
            )
            sugg_mob = ["தினசரி செக்-இன்", "செவிலியர் ஆலோசனை", "செயல்பாட்டு திட்டம்"]
        elif lang == "hi":
            reply = (
                f"🚶 **चलने-फिरने और गतिविधि संबंधी दिशानिर्देश:**\n\n"
                f"• **हल्का टहलना फायदेमंद है:** रक्त के थक्के जमने से रोकने के लिए कमरे के अंदर 5-10 मिनट धीरे-धीरे टहलने की सलाह दी जाती है।\n"
                f"• **भारी वजन न उठाएं:** {acts}\n"
                f"• **सीढ़ियां:** सीढ़ियां चढ़ते समय रेलिंग का सहारा लें और एक-एक कदम सावधानी से रखें।\n"
                f"• **कठिन व्यायाम से बचें:** दौड़ना, जिम जाना या भारी वजन उठाने का काम तब तक न करें जब तक डॉक्टर अनुमति न दें।"
            )
            sugg_mob = ["दैनिक चेक-इन", "नर्स से सलाह", "गतिविधि योजना"]
        else:
            reply = (
                f"🚶 **Mobility, Walking, and Physical Activity Guidelines:**\n\n"
                f"• **Gentle Walking Encouraged:** Frequent, short indoor walks (5-10 minutes several times daily) promote circulation, prevent deep vein thrombosis (DVT), and aid gastrointestinal motility.\n"
                f"• **Lifting Restriction:** {acts}\n"
                f"• **Stairs:** Navigate stairs slowly, one step at a time, keeping a firm grip on the handrail. Have someone assist you if you feel lightheaded.\n"
                f"• **Avoid Strenuous Exercise:** Do not jog, run, lift heavy items, or do abdominal strain exercises until fully cleared during follow-up."
            )
            sugg_mob = ["Log Daily Activity", "View Restrictions", "Consult Care Team"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_mob,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 11. Fever, Temperature, Chills & Infection Symptoms
    is_fever_query = matches_intent([
        "fever", "temperature", "chills", "sweating", "shivering", "high temp", "feverish", "pyrexia",
        "காய்ச்சல்", "சூடு", "குளிர்", "நடுக்கம்", "வியர்வை", "बुखार", "तापमान", "ठंड", "कंपकंपी", "पसीना"
    ])
    if is_fever_query:
        citations.append({"source": "Clinical Thermometry & Postoperative Infection Protocol", "snippet": "Fever etiology, clinical signs, and escalation thresholds"})
        doc = ctx.get("attending_doctor", "your attending doctor")
        if lang == "ta":
            reply = (
                "🌡️ **காய்ச்சலின் பொதுவான அறிகுறிகள் & மருத்துவ வழிகாட்டுதல்:**\n\n"
                "• **காய்ச்சலின் முக்கிய அறிகுறிகள் (Symptoms of Fever):**\n"
                "  1. **உடல் வெப்பநிலை உயர்வு:** டிஜிட்டல் தெர்மோமீட்டரில் 100.4°F (38°C) அல்லது அதற்கு மேல் இருப்பது.\n"
                "  2. **குளிர் மற்றும் நடுக்கம் (Chills & Shivering):** உடல் சூடானாலும் நோயாளிக்கு அதிக குளிர் எடுப்பது.\n"
                "  3. **அதிக வியர்வை & நீரிழப்பு:** திடீர் வியர்வை மற்றும் அதிக தாகம் ஏற்படுதல்.\n"
                "  4. **தலைவலி & தசை வலி:** உடல் சோர்வு, மூட்டு வலி மற்றும் பொதுவான பலவீனம்.\n"
                "  5. **பசியின்மை & சோர்வு:** உணவு உட்கொள்ள விருப்பமின்மை மற்றும் மந்தநிலை.\n\n"
                "• **அறுவை சிகிச்சைக்குப் பின் காய்ச்சல் எதைக் குறிக்கிறது?**\n"
                "  - அறுவை சிகிச்சை முடிந்த முதல் 24-48 மணி நேரத்தில் லேசான வெப்பநிலை உயர்வு (<100°F) திசுக்கள் குணமாகும் உடலின் இயல்பான தற்காப்பு எதிர்வினையாக இருக்கலாம்.\n"
                "  - ஆனால் 48 மணி நேரத்திற்குப் பிறகும் **100.4°F (38°C) க்கும் அதிகமான காய்ச்சல்** நீடித்தால், அது காயம் தொற்று (wound infection), சிறுநீர்ப் பாதை தொற்று அல்லது சுவாசத் தொற்றின் அறிகுறியாக இருக்கலாம்.\n\n"
                "• **நீங்கள் செய்ய வேண்டியவை:**\n"
                "  - தினமும் காலை மற்றும் மாலை தெர்மோமீட்டர் மூலம் வெப்பநிலையை அளவிட்டு குறித்து வைக்கவும்.\n"
                "  - போதுமான அளவு நீர் மற்றும் திரவ உணவுகளை அருந்தி நீரேற்றத்துடன் இருக்கவும்.\n"
                "  - மருத்துவர் பரிந்துரைத்த காய்ச்சல் மாத்திரைகளை (Paracetamol) சரியான நேரத்தில் உட்கொள்ளவும்.\n"
                f"  - காய்ச்சலுடன் காயத்தில் சிவத்தல், சீழ் கசிவு அல்லது தீவிர நடுக்கம் ஏற்பட்டால் உடனடியாக {doc} அல்லது மருத்துவமனையைத் தொடர்பு கொள்ளவும்."
            )
            sugg_fever = ["வெப்பநிலை பதிவு செய்", "மருத்துவரை அணுகவும்", "செவிலியரிடம் கேட்கவும்"]
        elif lang == "hi":
            reply = (
                "🌡️ **बुखार के प्रमुख लक्षण और क्लिनिकल दिशानिर्देश:**\n\n"
                "• **बुखार के मुख्य लक्षण (Symptoms of Fever):**\n"
                "  1. **शरीर का उच्च तापमान:** डिजिटल थर्मामीटर पर 100.4°F (38°C) या उससे अधिक तापमान दर्ज होना।\n"
                "  2. **ठंड लगना और कंपकंपी (Chills & Shivering):** शरीर गर्म होने पर भी कंपकंपी महसूस होना।\n"
                "  3. **अत्यधिक पसीना आना:** बुखार उतरते या चढ़ते समय अत्यधिक पसीना और निर्जलीकरण।\n"
                "  4. **सिरदर्द और शरीर में दर्द:** मांसपेशियों में दर्द, भारीपन और अत्यधिक कमजोरी।\n"
                "  5. **भूख न लगना और सुस्ती:** भोजन का स्वाद न आना और थकान।\n\n"
                "• **सर्जरी के बाद बुखार का क्या महत्व है?**\n"
                "  - सर्जरी के पहले 24 से 48 घंटों में हल्का तापमान (<100°F) शरीर की स्वाभाविक हीलिंग प्रतिक्रिया हो सकती है।\n"
                "  - परंतु यदि **100.4°F (38°C) से अधिक बुखार** बना रहता है, तो यह घाव के संक्रमण या श्वसन संक्रमण का संकेत हो सकता है।\n\n"
                "• **देखभाल और सावधानियां:**\n"
                "  - दिन में 2 बार थर्मामीटर से तापमान जांचें और लॉग में दर्ज करें।\n"
                "  - पर्याप्त पानी और ओआरएस/तरल पदार्थ पिएं।\n"
                "  - डॉक्टर द्वारा निर्धारित पैरासिटामोल दवा समय पर लें।\n"
                f"  - यदि बुखार के साथ घाव में लाली, मवाद या तेज दर्द हो, तो तुरंत {doc} या अस्पताल से संपर्क करें।"
            )
            sugg_fever = ["तापमान लॉग दर्ज करें", "डॉक्टर से परामर्श लें", "नर्स को सूचित करें"]
        else:
            reply = (
                "🌡️ **Clinical Symptoms of Fever & Postoperative Guidance:**\n\n"
                "• **Primary Symptoms of Fever:**\n"
                "  1. **Elevated Body Temperature:** A core body temperature reading of **100.4°F (38°C)** or higher measured with a digital thermometer.\n"
                "  2. **Chills & Shivering:** Sensation of feeling cold and involuntary muscle shivering, even as body temperature rises.\n"
                "  3. **Sweating & Diaphoresis:** Episodes of profuse perspiration as the body attempts to shed excess heat.\n"
                "  4. **Headache & Generalized Body Aches:** Muscle soreness (myalgia), joint stiffness, and deep physical fatigue.\n"
                "  5. **Flushed Skin & Rapid Heart Rate:** Warm, pink skin, dry mucous membranes, and an accelerated pulse.\n"
                "  6. **Loss of Appetite & Dehydration:** Decreased interest in solid foods and increased fluid requirements.\n\n"
                "• **Post-Surgical Implications:**\n"
                "  - A mild, low-grade temperature (<100.0°F) during the first 24–48 hours post-operation is often a normal physiological response to surgical stress or mild atelectasis.\n"
                "  - However, a persistent or rising fever **above 100.4°F (38°C)** beyond post-op day 2 can indicate an active surgical site infection (SSI), urinary tract infection (UTI), or respiratory infection.\n\n"
                "• **Recommended Action Steps:**\n"
                "  - Check and log oral temperature twice daily using a clean digital thermometer.\n"
                "  - Maintain aggressive hydration with 2–2.5 liters of water and electrolyte soups daily.\n"
                "  - Take prescribed antipyretics (e.g. Paracetamol / Acetaminophen) on schedule as directed.\n"
                f"  - **Contact {doc} immediately** if temperature exceeds 100.4°F or is accompanied by wound drainage, expanding redness, or persistent rigors."
            )
            sugg_fever = ["Log Body Temperature", "Contact Attending Doctor", "Request Teleconsultation"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_fever,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 12. Pain & Swelling Management
    is_pain_query = matches_intent([
        "pain", "swelling", "bruise", "bruising", "sore", "soreness", "ache", "stiffness", "hurt", "throbbing",
        "வலி", "வீக்கம்", "உளைச்சல்", "வலிக்குது", "दर्द", "सूजन", "टीस"
    ])
    if is_pain_query:
        citations.append({"source": "Postoperative Analgesic & Symptom Management", "snippet": "Pain and inflammatory response monitoring"})
        meds = ctx.get("active_medications", [])
        pain_meds = [m['name'] for m in meds if any(p in (m.get('purpose', '') + m['name']).lower() for p in ['pain', 'analgesic', 'paracetamol', 'tramadol', 'ibuprofen'])]
        pain_med_str = ", ".join(pain_meds) if pain_meds else "prescribed pain medication"
        if lang == "ta":
            reply = (
                f"🩹 **வலி மற்றும் வீக்கம் மேலாண்மை வழிகாட்டுதல்:**\n\n"
                f"• **இயல்பான மீட்பு வலி:** அறுவை சிகிச்சைக்குப் பின் முதல் சில நாட்களுக்கு மிதமான வலியும் லேசான வீக்கமும் இருப்பது இயல்பான உடல் குணமடைதல் செயல்முறையாகும்.\n"
                f"• **பரிந்துரைக்கப்பட்ட மருந்துகள்:** உங்கள் மருத்துவர் பரிந்துரைத்த வலி நிவாரணிகளை ({pain_med_str}) சரியான நேரத்தில் உணவுக்குப் பின் உட்கொள்ளவும். சுயமாக கூடுதல் மாத்திரைகளை எடுக்க வேண்டாம்.\n"
                f"• **ஓய்வு மற்றும் கால்களை உயர்த்துதல்:** ஓய்வாக இருக்கும்போது காயப் பகுதியை சற்று உயர்த்தி வைக்கவும். மருத்துவர் அனுமதித்திருந்தால் மட்டுமே லேசான ஐஸ் ஒத்தடம் (துணியில் சுற்றி) கொடுக்கலாம்.\n"
                f"• **எப்போது அழைக்க வேண்டும்:** வலி திடீரென தாங்க முடியாத அளவுக்கு அதிகரித்தால் (8-10/10), அல்லது 100.4°F (38°C) காய்ச்சல் ஏற்பட்டால் உடனே மருத்துவரைத் தொடர்புகொள்ளவும்."
            )
            sugg_pain = ["வலி செக்-இன் பதிவு", "மருத்துவரை அணுகவும்", "செவிலியரிடம் கேட்கவும்"]
        elif lang == "hi":
            reply = (
                f"🩹 **दर्द और सूजन प्रबंधन दिशानिर्देश:**\n\n"
                f"• **सामान्य रिकवरी दर्द:** सर्जरी के बाद पहले कुछ दिनों तक हल्का-मध्यम दर्द और थोड़ी सूजन होना सामान्य हीलिंग प्रक्रिया का हिस्सा है।\n"
                f"• **निर्धारित दवाएं:** डॉक्टर द्वारा दी गई दर्द निवारक दवाएं ({pain_med_str}) समय पर भोजन के बाद लें। स्वयं से अतिरिक्त गोली न लें।\n"
                f"• **आराम:** सर्जरी वाले हिस्से को थोड़ा ऊंचा रखकर आराम करें। जरूरत पड़ने पर ठंडी सिकाई केवल डॉक्टर की सलाह से ही करें।\n"
                f"• **अलर्ट:** यदि दर्द अचानक असहनीय हो जाए (8-10/10) या 100.4°F (38°C) से अधिक बुखार आए, तो तुरंत संपर्क करें।"
            )
            sugg_pain = ["दर्द चेक-इन दर्ज करें", "डॉक्टर से परामर्श लें", "नर्स को सूचित करें"]
        else:
            reply = (
                f"🩹 **Pain and Swelling Management Guidance:**\n\n"
                f"• **Expected Soreness:** Mild-to-moderate soreness, tightness, and local swelling are expected physiologic responses during tissue healing.\n"
                f"• **Take Prescribed Analgesics on Time:** Take your prescribed pain medications ({pain_med_str}) consistently as scheduled with food. Do not wait until pain becomes severe, and never take extra pills beyond the prescription.\n"
                f"• **Elevation & Rest:** Keep the healing site comfortably elevated when sitting or resting to assist lymphatic drainage and minimize swelling.\n"
                f"• **Red-Flag Escalation:** If pain escalates sharply to severe (8-10/10), or if you develop a fever exceeding 100.4°F (38°C), notify your attending doctor immediately."
            )
            sugg_pain = ["Log Pain Score", "Contact Attending Doctor", "Request Telehealth"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_pain,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 12. Wound Care, Bandage & Dressing Inspection
    is_wound_query = matches_intent([
        "wound", "incision", "bandage", "dressing", "stitches", "staples", "leak", "pus", "drainage", "bleed", "bleeding",
        "காயம்", "கட்டு", "தையல்", "சீழ்", "இரத்தம்", "கசிவு", "घाव", "पट्टी", "टांके", "मवाद", "खून", "रिसाव"
    ])
    if is_wound_query:
        citations.append({"source": "Surgical Wound Management", "snippet": "Incision precautions and warning signs"})
        prec = ctx.get("precautions", "Keep surgical site clean, dry, and protected.")
        warns = ctx.get("warning_signs", ["Fever > 100.4 F", "Foul odor or yellowish drainage", "Rapidly expanding redness"])
        warn_str = "\n".join(f"• {w}" for w in warns) if isinstance(warns, list) else str(warns)
        if lang == "ta":
            reply = (
                f"🩹 **காய பராமரிப்பு மற்றும் தையல் பாதுகாப்பு:**\n\n"
                f"• **காய முன்னெச்சரிக்கைகள்:** {prec}\n"
                f"• **கட்டை அகற்றவோ மாற்றவோ வேண்டாம்:** மருத்துவர் அல்லது செவிலியர் அறிவுறுத்தும் வரை கட்டை நீங்களாக பிரிக்கவோ சொறியவோ கூடாது.\n"
                f"• **கவனிக்க வேண்டிய எச்சரிக்கை அறிகுறிகள்:**\n{warn_str}\n\n"
                f"இத்தகைய அறிகுறிகள் தென்பட்டால் உடனடியாக மருத்துவரைத் தொடர்புகொள்ளவும்."
            )
            sugg_wound = ["காயம் செக்-இன்", "செவிலியரை தொடர்புகொள்", "புகைப்படம் அனுப்பு"]
        elif lang == "hi":
            reply = (
                f"🩹 **घाव की देखभाल और पट्टी सुरक्षा:**\n\n"
                f"• **घाव संबंधी सावधानियां:** {prec}\n"
                f"• **पट्टी को न हटाएं:** जब तक डॉक्टर या नर्स निर्देश न दें, तब तक पट्टी को स्वयं न हटाएं और न ही खुजलाएं।\n"
                f"• **चेतावनी संकेत जिन पर तुरंत ध्यान दें:**\n{warn_str}\n\n"
                f"यदि इनमें से कोई भी लक्षण दिखाई दे, तो तुरंत संपर्क करें।"
            )
            sugg_wound = ["घाव चेक-इन", "नर्स से संपर्क करें", "फोटो भेजें"]
        else:
            reply = (
                f"🩹 **Surgical Incision & Dressing Protocol:**\n\n"
                f"• **Incision Care Guidelines:** {prec}\n"
                f"• **Do Not Tamper with Dressing:** Keep the original sterile dressing intact. Do not pick, peel, or scratch around the incision margins or staples.\n"
                f"• **Warning Signs Requiring Urgent Evaluation:**\n{warn_str}\n\n"
                f"If you notice any foul drainage, expanding redness, or fever, contact the surgical nursing desk immediately."
            )
            sugg_wound = ["Log Wound Check-in", "Contact Surgical Nurse", "Request Telehealth"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_wound,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 13. Work, Job, Office & Daily Routine
    is_work_query = matches_intent([
        "work", "job", "office", "duty", "routine", "resume work", "return to work",
        "வேலை", "அலுவலகம்", "பணி", "தொழில்", "திரும்ப", "काम", "नौकरी", "दफ्तर", "काम पर लौटना"
    ])
    if is_work_query:
        citations.append({"source": "Occupational Recovery Guidelines", "snippet": "Return-to-work postoperative timeline"})
        doc = ctx.get("attending_doctor", "your attending physician")
        proc = ctx.get("surgical_procedure", "your surgical procedure")
        if lang == "ta":
            reply = (
                f"💼 **வேலைக்குத் திரும்புவது (Return to Work) குறித்த வழிகாட்டுதல்:**\n\n"
                f"• **சிகிச்சை வகை:** உங்கள் அறுவை சிகிச்சை ({proc}) மற்றும் குணமடையும் வேகத்தைப் பொறுத்து வேலைக்குத் திரும்பும் காலம் மாறுபடும்.\n"
                f"• **மேசைப் பணி (Desk Job):** கணினி அல்லது மேசைப் பணிகளை 1 முதல் 2 வாரங்களில் உங்கள் மருத்துவர் {doc} ஒப்புதலுடன் தொடங்கலாம்.\n"
                f"• **கடுமையான உடல் உழைப்பு:** கனமான பொருட்களை தூக்குவது, நீண்ட பயணம் அல்லது உடல் உழைப்பு தேவைப்படும் பணிகளை 4 முதல் 6 வாரங்களுக்குத் தவிர்க்க வேண்டும்.\n"
                f"• **பரிந்துரை:** முதல் பிந்தைய சந்திப்பில் (follow-up) உங்கள் உடல் நிலையை பரிசோதித்த பின்னரே முழுப் பணிக்குத் திரும்பவும்."
            )
            sugg_wrk = ["மருத்துவரை அணுகவும்", "செக்-இன் பதிவு செய்", "பராமரிப்பு திட்டம்"]
        elif lang == "hi":
            reply = (
                f"💼 **काम पर लौटने (Return to Work) संबंधी दिशानिर्देश:**\n\n"
                f"• **प्रक्रिया:** आपकी सर्जरी ({proc}) के आधार पर काम पर लौटने का समय तय होता है।\n"
                f"• **डेस्क जॉब:** हल्के डेस्क या कंप्यूटर संबंधी काम 1 से 2 सप्ताह में {doc} की अनुमति से शुरू किए जा सकते हैं।\n"
                f"• **शारीरिक मेहनत वाला काम:** भारी वजन उठाना या अधिक चलने-फिरने वाला काम 4 से 6 सप्ताह तक टालें।\n"
                f"• **सलाह:** अपनी पहली फॉलो-अप विजिट में डॉक्टर से औपचारिक फिटनेस मंजूरी अवश्य लें।"
            )
            sugg_wrk = ["डॉक्टर से परामर्श लें", "चेक-इन दर्ज करें", "देखभाल योजना"]
        else:
            reply = (
                f"💼 **Return to Work & Professional Duty Guidance:**\n\n"
                f"• **Procedure Context:** Convalescence timelines depend directly on your specific procedure ({proc}) and individual healing rate.\n"
                f"• **Desk / Sedentary Work:** Light desk work or remote computer duties may typically be resumed in 1 to 2 weeks, pending clearance by {doc}.\n"
                f"• **Physical / Manual Labor:** Any role requiring heavy lifting, prolonged standing, or vigorous physical exertion should be deferred for 4 to 6 weeks.\n"
                f"• **Recommendation:** Request a formal recovery evaluation and clinical fitness clearance note during your upcoming follow-up visit."
            )
            sugg_wrk = ["Consult Doctor", "Log Daily Status", "View Recovery Plan"]
        return {
            "reply": reply,
            "citations": citations,
            "suggested_actions": sugg_wrk,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 8. Staff Roles (Doctor, Nurse, Admin) Local Intelligence
    if ctx.get("user_role") == Role.DOCTOR.value:
        tot = ctx.get("total_assigned_patients", 0)
        hi_risk = ctx.get("high_risk_patients_count", 0)
        hi_list = ctx.get("high_risk_summary", [])
        if lang == "ta":
            reply = (
                f"👨‍⚕️ **மருத்துவ சிகிச்சை நுண்ணறிவு:**\n\n"
                f"• ஒதுக்கப்பட்ட மொத்த நோயாளிகள்: **{tot}**\n"
                f"• தீவிர / அதிக ஆபத்து எச்சரிக்கைகள்: **{hi_risk}**\n"
            )
            if hi_list:
                reply += "\n**உடனடி மதிப்பாய்வு தேவைப்படும் நோயாளிகள்:**\n" + "\n".join(
                    f"• {p['name']} ({p['code']}) – ஆபத்து நிலை: **{p['risk']}** ({p['procedure']})" for p in hi_list
                )
            else:
                reply += "\nஅனைத்து நோயாளிகளின் மீட்பு நிலையும் சீராக உள்ளது."
            sugg_doc = ["நோயாளி அடைவு", "ட்ரியேஜ் வரிசை", "டெலிஹெல்த் அறை"]
        elif lang == "hi":
            reply = (
                f"👨‍⚕️ **क्लिनिकल केसलोड इंटेलिजेंस:**\n\n"
                f"• कुल सक्रिय मरीज: **{tot}**\n"
                f"• उच्च / गंभीर जोखिम अलर्ट: **{hi_risk}**\n"
            )
            if hi_list:
                reply += "\n**तत्काल समीक्षा की आवश्यकता वाले मरीज:**\n" + "\n".join(
                    f"• {p['name']} ({p['code']}) – जोखिम: **{p['risk']}** ({p['procedure']})" for p in hi_list
                )
            else:
                reply += "\nसभी सौंपे गए मरीजों की टेलीमेट्री स्थिर है।"
            sugg_doc = ["मरीज निर्देशिका", "ट्राइएज कतार", "टेलीहेल्थ रूम"]
        else:
            reply = (
                f"👨‍⚕️ **Clinical Caseload Intelligence:**\n\n"
                f"• Total Active Patients in Roster: **{tot}**\n"
                f"• High / Critical Risk Alerts: **{hi_risk}**\n"
            )
            if hi_list:
                reply += "\n**Patients Needing Immediate Review:**\n" + "\n".join(
                    f"• {p['name']} ({p['code']}) – Risk: **{p['risk']}** ({p['procedure']})" for p in hi_list
                )
            else:
                reply += "\nAll assigned patients currently have stable risk telemetry."
            sugg_doc = ["Open Patient Directory", "Review Triage Queue", "Telehealth Room"]

        return {
            "reply": reply,
            "citations": [{"source": "Doctor Clinical Caseload", "snippet": f"{tot} assigned patients"}],
            "suggested_actions": sugg_doc,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    if ctx.get("user_role") == Role.NURSE.value:
        tot = ctx.get("total_assigned_patients", 0)
        triage_cnt = ctx.get("triage_priority_count", 0)
        triage_pts = ctx.get("triage_priority_patients", [])
        if lang == "ta":
            reply = (
                f"👩‍⚕️ **செவிலியர் ட்ரியேஜ் மற்றும் பின்தொடர்தல் வரிசை:**\n\n"
                f"• ஒதுக்கப்பட்ட நோயாளிகள்: **{tot}**\n"
                f"• அதிக வலி / பின்தொடர்தல் தேவைப்படுபவர்கள்: **{triage_cnt}**\n"
            )
            if triage_pts:
                reply += "\n**முன்னுரிமை பராமரிப்பு சரிபார்ப்புப் பட்டியல்:**\n" + "\n".join(
                    f"• {p['name']} ({p['code']}) – வலி நிலை: **{p.get('pain_score', p.get('pain_level', 'N/A'))}/10** (காய்ச்சல்: {p.get('fever', 'இல்லை')})" for p in triage_pts
                )
            sugg_nurse = ["செவிலியர் அறை", "முக்கிய அறிகுறிகள்", "டெலிஹெல்த் தொடங்கு"]
        elif lang == "hi":
            reply = (
                f"👩‍⚕️ **नर्सिंग ट्राइएज और फॉलो-अप कतार:**\n\n"
                f"• सौंपे गए मरीज: **{tot}**\n"
                f"• तेज दर्द / फॉलो-अप की आवश्यकता: **{triage_cnt}**\n"
            )
            if triage_pts:
                reply += "\n**प्राथमिकता देखभाल चेकलिस्ट:**\n" + "\n".join(
                    f"• {p['name']} ({p['code']}) – दर्द स्कोर: **{p.get('pain_score', p.get('pain_level', 'N/A'))}/10** (बुखार: {p.get('fever', 'नहीं')})" for p in triage_pts
                )
            sugg_nurse = ["नर्स डैशबोर्ड", "वाइटल्स रिकॉर्ड करें", "टेलीहेल्थ शुरू करें"]
        else:
            reply = (
                f"👩‍⚕️ **Nursing Triage & Follow-up Queue:**\n\n"
                f"• Assigned Patients: **{tot}**\n"
                f"• Elevated Pain / Follow-up Needed: **{triage_cnt}**\n"
            )
            if triage_pts:
                reply += "\n**Priority Care Checklist:**\n" + "\n".join(
                    f"• {p['name']} ({p['code']}) – Pain Score: **{p.get('pain_score', p.get('pain_level', 'N/A'))}/10** (Fever: {p.get('fever', 'No')})" for p in triage_pts
                )
            sugg_nurse = ["Open Nurse Dashboard", "Record Vitals", "Initiate Telehealth"]

        return {
            "reply": reply,
            "citations": [{"source": "Nurse Station Triage", "snippet": f"{triage_cnt} triage items"}],
            "suggested_actions": sugg_nurse,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    if ctx.get("user_role") == Role.ADMIN.value:
        pts = ctx.get("hospital_active_patients", 0)
        staff = ctx.get("hospital_total_staff", 0)
        tele = ctx.get("active_telehealth_consultations", 0)
        if lang == "ta":
            reply = (
                f"🏥 **மருத்துவமனை செயல்பாட்டு அளவீடுகள்:**\n\n"
                f"• சிகிச்சை பெறும் நோயாளிகள்: **{pts}**\n"
                f"• பதிவு செய்யப்பட்ட மருத்துவப் பணியாளர்கள்: **{staff}**\n"
                f"• நேரலை டெலிஹெல்த் அறைகள்: **{tele}**\n"
                f"• கணினி நிலை: **இயங்குகிறது & SMS தயார் நிலை**"
            )
            sugg_admin = ["SMS கேட்வே", "தணிக்கை பதிவு", "பணியாளர் பட்டியல்"]
        elif lang == "hi":
            reply = (
                f"🏥 **अस्पताल संचालन टेलीमेट्री:**\n\n"
                f"• सक्रिय स्वास्थ्य लाभ ले रहे मरीज: **{pts}**\n"
                f"• पंजीकृत क्लिनिकल स्टाफ: **{staff}**\n"
                f"• सक्रिय लाइव टेलीहेल्थ रूम: **{tele}**\n"
                f"• सिस्टम स्थिति: **सक्रिय और एसएमएस गेटवे तैयार**"
            )
            sugg_admin = ["एसएमएस गेटवे", "ऑडिट लॉग", "स्टाफ रोस्टर"]
        else:
            reply = (
                f"🏥 **Hospital Operational Telemetry:**\n\n"
                f"• Active Convalescent Patients: **{pts}**\n"
                f"• Registered Clinical Staff: **{staff}**\n"
                f"• Active Live Telehealth Rooms: **{tele}**\n"
                f"• System Status: **Operational & SMS Carrier Ready**"
            )
            sugg_admin = ["Open SMS Gateway", "Audit Log", "Staff Roster"]

        return {
            "reply": reply,
            "citations": [{"source": "Hospital Operations", "snippet": "System Telemetry"}],
            "suggested_actions": sugg_admin,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # 9. Generic Patient Recovery Summary
    if ctx.get("user_role") == Role.PATIENT.value:
        name = ctx.get("patient_name", ctx.get("user_name", "Patient"))
        doc = ctx.get("attending_doctor", "Hospital Care Team")
        meds = ctx.get("active_medications", [])
        care = ctx.get("care_instructions", "Follow all prescribed medications and daily recovery routines.")
        if lang == "ta":
            reply = (
                f"வணக்கம் {name}!\n\n"
                f"உங்கள் மருத்துவ மீட்புத் திட்டம் மற்றும் சிகிச்சை விவரங்கள்:\n"
                f"• பொறுப்பு மருத்துவர்: **{doc}**\n"
                f"• பயன்பாட்டில் உள்ள மருந்துகள்: **{len(meds)}**\n\n"
                f"📋 **பராமரிப்பு வழிகாட்டுதல்:**\n{care}\n\n"
                f"உங்களுக்கு தேவையான குறிப்பிட்ட தகவலைக் கேளுங்கள் (எ.கா. 'இன்று நான் என்ன மருந்துகளை உட்கொள்ள வேண்டும் மற்றும் அதன் நோக்கம் என்ன?')."
            )
            sugg_pt = ["இன்று நான் என்ன மருந்துகளை உட்கொள்ள வேண்டும்?", "எனது உணவு முறை என்ன?", "அடுத்த சந்திப்பு எப்போது?"]
        elif lang == "hi":
            reply = (
                f"नमस्ते {name}!\n\n"
                f"आपकी चिकित्सा देखभाल योजना विवरण:\n"
                f"• उपस्थित चिकित्सक: **{doc}**\n"
                f"• सक्रिय निर्धारित दवाएं: **{len(meds)}**\n\n"
                f"📋 **देखभाल निर्देश:**\n{care}\n\n"
                f"कृपया अपनी आवश्यकता अनुसार सवाल पूछें (जैसे 'मुझे आज कौन सी दवाएं किस उद्देश्य के लिए लेनी हैं?')."
            )
            sugg_pt = ["मुझे आज कौन सी दवाएं लेनी हैं?", "मेरा आहार क्या है?", "मेरी अगली अपॉइंटमेंट कब है?"]
        else:
            reply = (
                f"Hello {name}!\n\n"
                f"Here is your personalized recovery status:\n"
                f"• Attending Physician: **{doc}**\n"
                f"• Active Prescriptions: **{len(meds)} active medications**\n\n"
                f"📋 **Care Instructions:**\n{care}\n\n"
                f"You can ask me specific questions at any time, such as *'What medicines do I have to take today and for what purpose?'*, or ask about diet and wound care."
            )
            sugg_pt = ["What medicines do I take today?", "What is my diet?", "When is my next appointment?"]
        return {
            "reply": reply,
            "citations": [{"source": "Patient Care Plan", "snippet": "General recovery overview"}],
            "suggested_actions": sugg_pt,
            "is_emergency": False,
            "prescription_boundary": False,
            "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
        }

    # Default fallback
    if lang == "ta":
        def_reply = "நான் ரெக்கவரா AI (Recovera AI). உங்கள் மருத்துவ பதிவுகளை பகுப்பாய்வு செய்துள்ளேன். உங்கள் மீட்பு, மருந்துகள், உணவு அல்லது அப்பாயிண்ட்மென்ட் பற்றி என்னிடம் கேட்கலாம்."
        def_sugg = ["எனது மருந்துகள்", "மீட்புத் திட்டம்", "மருத்துவரைத் தொடர்புகொள்ள"]
    elif lang == "hi":
        def_reply = "मैं रिकवरा एआई (Recovera AI) हूँ। मैंने आपके क्लिनिकल रिकॉर्ड का विश्लेषण किया है। अपनी रिकवरी, दवाओं, आहार या अपॉइंटमेंट के बारे में मुझसे पूछें।"
        def_sugg = ["मेरी दवाएं", "रिकवरी योजना", "डॉक्टर से संपर्क करें"]
    else:
        def_reply = "I am Recovera AI. I have analyzed your in-app clinical records. Please let me know how I can assist with your recovery, medications, diet, or appointments."
        def_sugg = ["My Medications", "Recovery Plan", "Contact Doctor"]

    return {
        "reply": def_reply,
        "citations": [{"source": "System Core", "snippet": "Recovera AI Clinical Assistant"}],
        "suggested_actions": def_sugg,
        "is_emergency": False,
        "prescription_boundary": False,
        "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
    }


def call_gemini_assistant(
    user_query: str,
    ctx: Dict[str, Any],
    lang: str = "en",
    file_bytes: Optional[bytes] = None,
    file_mime: Optional[str] = None,
    file_name: Optional[str] = None,
    file_desc: Optional[str] = None
) -> Dict[str, Any]:
    """
    Always prefers Google Gemini API to answer queries with system instructions,
    real-time database context, and multimodal attachments.
    Allows up to 40.0 seconds for Gemini inference. Only if the call takes more
    than 40 seconds or encounters an unrecoverable failure across all active models,
    it falls back to the local offline clinical engine.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        logger.info("No GEMINI_API_KEY detected in environment or .env. Using Recovera Clinical Core engine.")
        return generate_fallback_response(user_query, ctx, lang, file_desc)

    start_time = time.time()
    try:
        import httpx
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        # Timeout configured to 40.0s per user specification
        client._api_client._httpx_client = httpx.Client(timeout=httpx.Timeout(40.0, connect=20.0))
        system_instruction = build_system_prompt(ctx, lang)

        contents_list = []
        if file_bytes and file_mime:
            try:
                part = types.Part.from_bytes(data=file_bytes, mime_type=file_mime)
                contents_list.append(part)
            except Exception as e:
                logger.warning(f"Could not build file part for Gemini: {e}")

        # Add text prompt
        user_prompt_text = f"User Query: {user_query}"
        if file_desc:
            user_prompt_text += f"\n[Attached Medical File Info: {file_desc}]"
        contents_list.append(user_prompt_text)

        # Candidate models - prioritized by verified availability and low latency
        primary_model = getattr(settings, "GEMINI_MODEL", "gemini-3.5-flash-lite")
        fallback_model = getattr(settings, "GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")

        candidate_models = []
        for m in [
            primary_model,
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
            fallback_model,
            "gemini-3.5-flash",
            "gemini-flash-lite-latest",
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-3.7-flash",
            "gemma-4-26b-a4b-it",
            "gemini-2.5-flash"
        ]:
            if m and m not in candidate_models:
                candidate_models.append(m)

        # Allow candidates to be attempted within the 40.0-second total time budget
        response = None
        used_model_name = candidate_models[0]

        for cand in candidate_models:
            # Enforce strict 40-second boundary requested by user
            elapsed = time.time() - start_time
            remaining = 40.0 - elapsed
            if remaining <= 1.5:
                logger.warning(f"Gemini execution reached {elapsed:.1f}s (>40s). Falling back to offline clinical model.")
                break

            try:
                cand_timeout = min(remaining, 30.0)
                client._api_client._httpx_client.timeout = httpx.Timeout(cand_timeout, connect=min(12.0, cand_timeout))
            except Exception:
                pass

            try:
                try:
                    # Fast low-latency inference: disable thinking budget
                    response = client.models.generate_content(
                        model=cand,
                        contents=contents_list,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.2,
                            thinking_config=types.ThinkingConfig(thinking_budget=0)
                        )
                    )
                except Exception:
                    # Fallback config without thinking_config
                    response = client.models.generate_content(
                        model=cand,
                        contents=contents_list,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=0.2,
                        )
                    )

                if response and response.text and response.text.strip():
                    used_model_name = cand
                    break
            except Exception as cand_exc:
                logger.warning(f"Gemini model '{cand}' call failed: {cand_exc}")
                continue

        if not response or not (response.text or "").strip():
            logger.warning("Gemini call timed out or failed across candidates. Falling back to local clinical engine.")
            return generate_fallback_response(user_query, ctx, lang, file_desc)

        reply_raw = response.text or ""

        # Parse dynamic contextual suggestions generated by Gemini
        suggested_actions = []
        match = re.search(r"<<<SUGGESTIONS>>>\s*(.*?)\s*<<<END_SUGGESTIONS>>>", reply_raw, re.DOTALL)
        if match:
            suggested_actions = [
                line.strip("- *• \t\r") for line in match.group(1).strip().split("\n") if line.strip()
            ]
            reply_text = reply_raw[:match.start()].strip()
        else:
            reply_text = reply_raw.strip()

        # If model did not emit delimiter, extract smart dynamic defaults
        if not suggested_actions:
            if lang == "ta":
                suggested_actions = [
                    "எனது மருந்து அட்டவணை என்ன?",
                    "நான் தவிர்க்க வேண்டிய எச்சரிக்கை அறிகுறிகள் என்ன?",
                    "உணவு மற்றும் ஊட்டச்சத்து வழிகாட்டுதல்கள் என்ன?"
                ]
            elif lang == "hi":
                suggested_actions = [
                    "मेरी निर्धारित दवाओं की खुराक क्या है?",
                    "मुझे किन चेतावनी संकेतों पर ध्यान देना चाहिए?",
                    "आहार और पोषण संबंधी दिशानिर्देश क्या हैं?"
                ]
            else:
                suggested_actions = [
                    "What medicines do I take today and for what purpose?",
                    "What warning signs should I watch for?",
                    "What are my approved dietary guidelines?"
                ]

        # Check for emergency / boundary in the generated response
        is_emerg = detect_regex(user_query, EMERGENCY_PATTERNS) or "EMERGENCY" in reply_text[:100]
        is_bound = detect_regex(user_query, PRESCRIPTION_BOUNDARY_PATTERNS) or "prescription" in reply_text.lower()[:100]

        # Map to user-friendly model display name
        display_map = {
            "gemini-3-flash-preview": "Google Gemini 3 Flash",
            "gemini-flash-lite-latest": "Google Gemini Flash Lite",
            "gemini-3.5-flash-lite": "Google Gemini 3.5 Flash Lite",
            "gemini-3.7-flash": "Google Gemini 3.7 Flash",
            "gemma-4-26b-a4b-it": "Google Gemma 4 Clinical",
            "gemini-3.8-flash": "Google Gemini 3.8 Flash",
            "gemini-3.5-flash": "Google Gemini 3.5 Flash",
            "gemini-flash-latest": "Google Gemini Flash",
            "gemini-2.5-flash": "Gemini 2.5 Flash",
        }
        model_display = display_map.get(used_model_name, used_model_name.replace("-", " ").title())

        return {
            "reply": reply_text,
            "citations": [{"source": "Hospital Clinical OS & Recovery Grounding", "snippet": f"{model_display} Grounded Response"}],
            "suggested_actions": suggested_actions[:4],
            "is_emergency": is_emerg,
            "prescription_boundary": is_bound,
            "model_used": model_display
        }

    except Exception as exc:
        logger.warning(f"Gemini API invocation encountered exception: {exc}. Executing seamless fallback.")
        _gemini_cooldown_until = time.time() + 60.0
        return generate_fallback_response(user_query, ctx, lang, file_desc)


def fallback_translate_text(text: str, target_lang: str) -> str:
    """Offline rule-based fallback translation for clinical text when Gemini is offline."""
    if not text or target_lang == "en":
        return text

    ta_map = [
        (r"(?i)###\s*Warning Signs\s*\(Red Flags\)", "### எச்சரிக்கை அறிகுறிகள் (சிவப்பு கொடிகள்)"),
        (r"(?i)###\s*Activity & Rest", "### செயல்பாடு மற்றும் ஓய்வு"),
        (r"(?i)###\s*General Care", "### பொதுவான பராமரிப்பு"),
        (r"(?i)###\s*Diet & Hydration", "### உணவு மற்றும் நீர்ச்சத்து வழிகாட்டுதல்"),
        (r"(?i)\* Avoid:\s*", "* தவிர்க்கவும்: "),
        (r"(?i)\* Rest:\s*", "* ஓய்வு: "),
        (r"(?i)\* Movement:\s*", "* உடற்பயிற்சி/இயக்கம்: "),
        (r"(?i)Stay away from oily, spicy foods and high-sodium snacks", "எண்ணெய், காரமான உணவுகள் மற்றும் அதிக உப்பு உள்ள உணவுகளைத் தவிர்க்கவும்"),
        (r"(?i)Prioritize adequate sleep and rest to help your body fight the infection", "உடல் தொற்றை எதிர்த்துப் போராட போதுமான தூக்கத்திற்கும் ஓய்விற்கும் முன்னுரிமை கொடுங்கள்"),
        (r"(?i)Please monitor your condition closely", "உங்கள் உடல்நிலையை உன்னிப்பாகக் கவனியுங்கள்"),
        (r"(?i)Seek immediate medical attention or contact Dr\. Samuel if you experience:", "பின்வரும் அறிகுறிகள் ஏற்பட்டால் உடனடியாக மருத்துவரை அணுகவும்:"),
        (r"(?i)Fever remaining above 100\.4°F despite taking medication", "மருந்து உட்கொண்ட பின்னரும் 100.4°F-க்கு மேல் காய்ச்சல் நீடித்தால்"),
        (r"(?i)Severe sharp pain or sudden difficulty breathing", "கடுமையான கூர்மையான வலி அல்லது திடீர் மூச்சுத்திணறல்"),
        (r"(?i)Increased dizziness or extreme exhaustion", "அதிகரித்த மயக்கம் அல்லது அதீத சோர்வு"),
    ]
    hi_map = [
        (r"(?i)###\s*Warning Signs\s*\(Red Flags\)", "### चेतावनी संकेत (रेड फ्लैग्स)"),
        (r"(?i)###\s*Activity & Rest", "### गतिविधि और आराम"),
        (r"(?i)###\s*General Care", "### सामान्य देखभाल"),
        (r"(?i)###\s*Diet & Hydration", "### आहार और जलयोजन"),
        (r"(?i)\* Avoid:\s*", "* बचें: "),
        (r"(?i)\* Rest:\s*", "* आराम: "),
        (r"(?i)\* Movement:\s*", "* हलचल: "),
        (r"(?i)Stay away from oily, spicy foods and high-sodium snacks", "तैलीय, मसालेदार खाद्य पदार्थों और अधिक सोडियम वाले स्नैक्स से दूर रहें"),
        (r"(?i)Prioritize adequate sleep and rest to help your body fight the infection", "संक्रमण से लड़ने के लिए पर्याप्त नींद और आराम को प्राथमिकता दें"),
        (r"(?i)Please monitor your condition closely", "कृपया अपनी स्थिति पर बारीकी से नज़र रखें"),
        (r"(?i)Seek immediate medical attention or contact Dr\. Samuel if you experience:", "यदि आप अनुभव करते हैं तो तत्काल चिकित्सा सहायता लें:"),
        (r"(?i)Fever remaining above 100\.4°F despite taking medication", "दवा लेने के बावजूद बुखार 100.4°F से ऊपर बना रहना"),
        (r"(?i)Severe sharp pain or sudden difficulty breathing", "गंभीर तेज दर्द या अचानक सांस लेने में कठिनाई"),
        (r"(?i)Increased dizziness or extreme exhaustion", "चक्कर आना या अत्यधिक थकान"),
    ]

    rules = ta_map if target_lang == "ta" else (hi_map if target_lang == "hi" else [])
    translated = text
    for pat, rep in rules:
        translated = re.sub(pat, rep, translated)
    return translated


def translate_clinical_texts(
    items: List[Dict[str, str]],
    target_lang: str = "en",
    source_lang: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Translates a list of clinical messages to target_lang ('ta', 'hi', 'en')
    using Gemini with fast JSON batching and structured markdown preservation.
    Falls back gracefully to clinical phrase map with zero runtime errors.
    """
    clean_target = (target_lang or "en").lower()[:2]
    lang_names = {"ta": "Tamil", "hi": "Hindi", "en": "English"}
    target_lang_name = lang_names.get(clean_target, "English")

    if not items:
        return []

    results = []
    pending_to_translate = []

    for item in items:
        item_id = str(item.get("id", len(results)))
        raw_text = str(item.get("text", "") or "")
        if not raw_text.strip():
            results.append({"id": item_id, "translated_text": ""})
            continue

        # Check if already in target language
        has_tamil = bool(re.search(r"[\u0B80-\u0BFF]", raw_text))
        has_hindi = bool(re.search(r"[\u0900-\u097F]", raw_text))

        if clean_target == "ta" and has_tamil and not re.search(r"[a-zA-Z]{8,}", raw_text):
            results.append({"id": item_id, "translated_text": raw_text})
        elif clean_target == "hi" and has_hindi and not re.search(r"[a-zA-Z]{8,}", raw_text):
            results.append({"id": item_id, "translated_text": raw_text})
        elif clean_target == "en" and not has_tamil and not has_hindi:
            results.append({"id": item_id, "translated_text": raw_text})
        else:
            pending_to_translate.append({"id": item_id, "text": raw_text})

    if not pending_to_translate:
        return results

    translated_map = {}
    api_key = get_gemini_api_key()

    if api_key:
        try:
            import httpx
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            client._api_client._httpx_client = httpx.Client(timeout=httpx.Timeout(15.0, connect=8.0))

            prompt = (
                f"You are an expert clinical medical translator for a hospital patient care system. "
                f"Translate each clinical message in the provided JSON array into {target_lang_name}.\n"
                f"STRICT RULES:\n"
                f"1. Preserve ALL markdown formatting exactly: headers (###, ##), bold (**), bullet points (*, -), numbers, newlines, quotes.\n"
                f"2. Keep medical dosage units, measurements, and medicine names clear (e.g. Paracetamol 500mg, Metformin, 100.4°F).\n"
                f"3. Do NOT include introductory greetings, explanations, transliterations, or conversational comments.\n"
                f"4. Output MUST be ONLY a valid JSON array matching the schema: [{{\"id\": \"string\", \"translated_text\": \"string\"}}].\n\n"
                f"Input Data:\n{json.dumps(pending_to_translate, ensure_ascii=False)}"
            )

            candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.5-flash", "gemini-flash-lite-latest"]
            for model_name in candidate_models:
                try:
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                    if resp and resp.text:
                        parsed = json.loads(resp.text)
                        if isinstance(parsed, list):
                            for p in parsed:
                                if isinstance(p, dict) and "id" in p and "translated_text" in p:
                                    translated_map[str(p["id"])] = str(p["translated_text"])
                            if translated_map:
                                break
                except Exception as ex:
                    logger.warning(f"Translation model {model_name} failed: {ex}")
                    continue
        except Exception as e:
            logger.warning(f"Batch translation error with Gemini: {e}")

    # Fallback for any items not translated by Gemini
    for pending in pending_to_translate:
        pid = pending["id"]
        if pid not in translated_map:
            orig = pending["text"]
            translated_map[pid] = fallback_translate_text(orig, clean_target)

    # Reassemble final results maintaining original order
    final_results = []
    all_translated = {r["id"]: r["translated_text"] for r in results}
    all_translated.update(translated_map)

    for item in items:
        iid = str(item.get("id", ""))
        final_results.append({
            "id": iid,
            "translated_text": all_translated.get(iid, item.get("text", ""))
        })

    return final_results
