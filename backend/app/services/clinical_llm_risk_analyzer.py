"""
Recovera AI – Advanced Clinical LLM Risk Intelligence Engine
Integrates Google Gemini API (gemini-3.5-flash-lite / gemini-3.8-flash) to evaluate patient risk
by deeply analyzing:
1. Patient's diagnosed condition (Dengue, Acute Fever, Post-op, etc.)
2. Medical History (chronic diseases, prior surgeries, long-term medications, allergies)
3. Medication Intakes (adherence, specific missed/delayed critical doses)
4. Granular Daily Telemetry (pain levels, anatomical sites, fever, sleep, mobility)
5. Custom Free-Text Patient Notes / Messages (sentiment, infection signals, acute distress)

Outputs dual-tier clinical explainability:
- Patient-Facing Summary: Simple, empathetic, plain-language abstract explanation ("why").
- Doctor/Nurse Summary: Comprehensive clinical synthesis, comorbidity impact, symptom dynamics,
  custom note NLP flags, and concrete prioritized interventions.
"""

import os
import json
import logging
import datetime as dt
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.core.config import settings
from app.models.patient import Patient
from app.models.medication import Medication, MedicationOccurrence
from app.models.checkin import DailyCheckIn
from app.core.constants import OccurrenceState
from app.services.recovery_ai_service import get_gemini_api_key

logger = logging.getLogger("recovera.llm_risk_analyzer")


def _gather_patient_clinical_dossier(
    db: Session,
    patient_id: str,
    window_days: int = 7
) -> Dict[str, Any]:
    """Compiles comprehensive clinical profile, medical history, telemetry, and custom messages."""
    patient = db.get(Patient, patient_id)
    if not patient:
        raise ValueError(f"Patient {patient_id} not found.")

    now = dt.datetime.utcnow()
    window_start = now - dt.timedelta(days=window_days)

    # 1. Demographics & Active Condition
    p_name = patient.user.name if patient.user else patient.patient_code
    age = None
    if patient.dob:
        today_d = dt.date.today()
        age = today_d.year - patient.dob.year - ((today_d.month, today_d.day) < (patient.dob.month, patient.dob.day))

    active_condition = patient.active_condition or "General Post-Discharge Convalescence"
    condition_stage = patient.condition_stage or "active"

    # 2. Medical History & Chronic Baseline
    hist_raw = patient.history or {}
    prior_conditions = hist_raw.get("conditions", [])
    past_meds = hist_raw.get("past_medications", [])
    past_surgeries = hist_raw.get("past_treatments", [])
    past_hosp = hist_raw.get("past_hospitalizations", [])
    history_notes = hist_raw.get("notes", "")

    # 3. Medications & Specific Missed Doses
    active_meds = db.scalars(
        select(Medication).where(Medication.patient_id == patient_id, Medication.status == "active")
    ).all()

    meds_info = [
        {
            "name": m.name,
            "strength": m.strength or "",
            "dose": m.dose or "",
            "period": getattr(m, "period", "1-0-1"),
            "intake_type": getattr(m, "intake_type", "After Food"),
            "purpose": getattr(m, "purpose", "Clinical Recovery")
        }
        for m in active_meds
    ]

    occurrences = db.scalars(
        select(MedicationOccurrence)
        .where(
            MedicationOccurrence.patient_id == patient_id,
            MedicationOccurrence.scheduled_at >= window_start,
            MedicationOccurrence.scheduled_at <= now,
            MedicationOccurrence.state != OccurrenceState.CANCELLED.value
        )
        .order_by(desc(MedicationOccurrence.scheduled_at))
    ).all()

    total_doses = len(occurrences)
    taken_doses = sum(1 for o in occurrences if o.state == OccurrenceState.TAKEN.value)
    missed_doses = sum(1 for o in occurrences if o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value))

    missed_med_details = []
    for o in occurrences:
        if o.state in (OccurrenceState.MISSED.value, OccurrenceState.NOT_TAKEN.value):
            med_obj = db.get(Medication, o.medication_id) if o.medication_id else None
            missed_med_details.append({
                "medicine_name": med_obj.name if med_obj else "Prescribed Medication",
                "scheduled_time": o.scheduled_at.strftime("%Y-%m-%d %H:%M") if o.scheduled_at else "Past",
                "strength": med_obj.strength if med_obj else "",
                "purpose": getattr(med_obj, "purpose", "Unspecified")
            })

    adherence_pct = round((taken_doses / total_doses) * 100, 1) if total_doses > 0 else 100.0

    # 4. Daily Check-ins & Custom Verbatim Messages
    checkins = db.scalars(
        select(DailyCheckIn)
        .where(DailyCheckIn.patient_id == patient_id, DailyCheckIn.checkin_date >= window_start.date())
        .order_by(desc(DailyCheckIn.checkin_date))
    ).all()

    checkin_details = []
    custom_messages = []
    pains = []
    fevers = []

    for ci in checkins:
        p_lvl = ci.pain_level if ci.pain_level is not None else 0
        pains.append(p_lvl)
        if ci.fever:
            fevers.append(True)

        note_txt = (ci.notes or "").strip()
        if note_txt:
            custom_messages.append({
                "date": ci.checkin_date.isoformat(),
                "pain": p_lvl,
                "note": note_txt
            })

        checkin_details.append({
            "date": ci.checkin_date.isoformat(),
            "pain_level": p_lvl,
            "pain_location": ci.pain_location or "General",
            "fever": bool(ci.fever),
            "cough": bool(ci.cough),
            "sleep_hours": ci.sleep_hours,
            "activity_level": ci.activity_level or "normal",
            "notes": note_txt
        })

    max_pain = max(pains, default=0)
    has_fever = any(fevers)

    # Active telemetry in the last 48 hours / most recent submission
    today_d = now.date()
    recent_checkins = [ci for ci in checkins if ci.checkin_date >= (today_d - dt.timedelta(days=2))]
    if not recent_checkins and checkins:
        recent_checkins = [checkins[0]]
    recent_max_pain = max((ci.pain_level for ci in recent_checkins if ci.pain_level is not None), default=0)
    has_active_fever = any(ci.fever for ci in recent_checkins)

    return {
        "patient_id": patient_id,
        "name": p_name,
        "age": age,
        "gender": patient.gender or "unspecified",
        "active_condition": active_condition,
        "condition_stage": condition_stage,
        "medical_history": {
            "chronic_conditions": prior_conditions,
            "past_medications": past_meds,
            "past_surgeries": past_surgeries,
            "past_hospitalizations": past_hosp,
            "history_notes": history_notes
        },
        "medication_telemetry": {
            "active_medications": meds_info,
            "total_scheduled_doses": total_doses,
            "taken_doses": taken_doses,
            "missed_doses_count": missed_doses,
            "adherence_pct": adherence_pct,
            "missed_medication_details": missed_med_details[:5]
        },
        "symptom_telemetry": {
            "submissions_count": len(checkins),
            "max_pain": max_pain,
            "recent_max_pain": recent_max_pain,
            "recent_pains": pains[:4],
            "has_active_fever": has_active_fever,
            "prior_fever_resolved": (has_fever and not has_active_fever),
            "checkins": checkin_details[:5]
        },
        "custom_patient_messages": custom_messages
    }


def _execute_clinical_rule_fallback(
    dossier: Dict[str, Any],
    baseline_level: str,
    baseline_reasons: List[str]
) -> Dict[str, Any]:
    """Offline clinical reasoning engine if Gemini API is unreachable."""
    cond = dossier["active_condition"]
    stage = dossier["condition_stage"]
    hist = dossier["medical_history"]
    chronic = hist.get("chronic_conditions", [])
    meds = dossier["medication_telemetry"]
    symp = dossier["symptom_telemetry"]
    custom_msgs = dossier["custom_patient_messages"]

    # Analyze chronic comorbidity impact
    chronic_str = ", ".join(chronic) if chronic else "None documented"
    comorbidity_hazard = "low"
    if any(c.lower() in ("diabetes", "hypertension", "heart disease", "asthma", "ckd", "renal") for c in chronic):
        comorbidity_hazard = "elevated"

    # Analyze custom messages for clinical signals
    extracted_flags = []
    for msg in custom_msgs:
        t = msg["note"].lower()
        if "fever" in t or "temp" in t:
            extracted_flags.append(f"Temperature elevation reported: \"{msg['note']}\"")
        if "dizzy" in t or "faint" in t:
            extracted_flags.append(f"Neurological / Hemodynamic warning: \"{msg['note']}\"")
        if "pus" in t or "discharge" in t or "leak" in t:
            extracted_flags.append(f"Surgical site infection marker: \"{msg['note']}\"")
        if "rash" in t or "spot" in t or "bleed" in t:
            extracted_flags.append(f"Hemorrhagic / Vascular manifestation: \"{msg['note']}\"")
        if "pain" in t or "ache" in t or "hurts" in t:
            extracted_flags.append(f"Subjective pain escalation: \"{msg['note']}\"")

    # Determine risk level with single-day lapse guardrails
    final_lvl = baseline_level.lower()
    recent_pain = symp.get("recent_max_pain", symp["max_pain"])
    is_acute_emergency = bool(extracted_flags and any("infection" in f or "Hemorrhagic" in f or "Hemodynamic" in f for f in extracted_flags))

    if is_acute_emergency:
        final_lvl = "red"
    elif comorbidity_hazard == "elevated" and (recent_pain >= 7 or symp["has_active_fever"]) and not (meds["missed_doses_count"] <= 1 and not symp["has_active_fever"]):
        final_lvl = "red"
    elif recent_pain >= 5 or meds["adherence_pct"] < 80.0 or extracted_flags or symp.get("prior_fever_resolved"):
        if final_lvl != "red":
            final_lvl = "yellow"

    # Enforce strict single-day lapse cap: missing intake or check-in for 1 day cannot be red without acute emergency
    if final_lvl == "red" and not is_acute_emergency and not symp["has_active_fever"] and recent_pain < 8:
        final_lvl = "yellow"

    # Dual-tier summaries
    if final_lvl == "red":
        patient_summary = (
            f"Your care team has flagged your recovery for priority review due to reported discomfort "
            f"({symp['max_pain']}/10) and symptoms related to your {cond}. Please rest and reach out to your doctor or nursing team for prompt advice."
        )
        clinical_assessment = (
            f"Patient with {cond} ({stage}) and pre-existing {chronic_str} exhibits elevated clinical risk. "
            f"Telemetry indicates peak pain of {symp['max_pain']}/10, fever status: {symp['has_active_fever']}, and {meds['missed_doses_count']} missed doses."
        )
    elif final_lvl == "yellow":
        patient_summary = (
            f"Your recovery is generally on track, but your care team noted a slight increase in discomfort or a missed medication. "
            f"Please ensure you stay hydrated, take your prescribed doses, and log your evening check-in."
        )
        clinical_assessment = (
            f"Patient with {cond} shows moderate recovery variance. Medication compliance is {meds['adherence_pct']}%, "
            f"with pain telemetry averaging {symp['max_pain']}/10. Pre-existing conditions ({chronic_str}) require vigilant nursing oversight."
        )
    else:
        patient_summary = (
            f"Your recovery progression for {cond} is stable and on track. "
            f"All vital signs and medication schedules are within expected targets. Keep up the great progress!"
        )
        clinical_assessment = (
            f"Patient is demonstrating optimal physiological recovery for {cond}. "
            f"Medication adherence is {meds['adherence_pct']}%, vitals are stable, and pain is well-controlled."
        )

    return {
        "ai_risk_level": final_lvl,
        "clinical_score": 85 if final_lvl == "red" else (50 if final_lvl == "yellow" else 15),
        "patient_facing_summary": patient_summary,
        "doctor_nurse_summary": {
            "clinical_assessment": clinical_assessment,
            "medication_impact": f"Adherence is {meds['adherence_pct']}% across {meds['total_scheduled_doses']} tracked doses. {len(meds['missed_medication_details'])} specific missed dose events recorded.",
            "symptom_trajectory_analysis": f"Maximum pain VAS {symp['max_pain']}/10. Active fever flag: {symp['has_active_fever']}. Telemetry based on {symp['submissions_count']} daily reports.",
            "custom_notes_sentiment_flags": extracted_flags or ["No acute distress markers identified in patient notes."],
            "recommended_clinical_interventions": [
                f"Verify {cond} symptom trajectory and assess vital signs" if final_lvl == "red" else f"Routine nursing follow-up for {cond}",
                "Review medication adherence log with patient and confirm dose intake",
                f"Monitor comorbidity stability ({chronic_str})" if chronic else "Encourage continuous daily telemetry logging"
            ]
        },
        "medical_history_impact": f"Pre-existing {chronic_str} {'significantly compounds recovery risk' if comorbidity_hazard == 'elevated' else 'is currently well-tolerated'}.",
        "custom_message_analysis": f"Patient submitted {len(custom_msgs)} custom clinical notes in the window.",
        "model_used": "Recovera Clinical Core (Offline-Safe Engine)"
    }


def analyze_patient_risk_with_ai(
    db: Session,
    patient_id: str,
    baseline_level: str = "green",
    baseline_reasons: Optional[List[str]] = None,
    window_days: int = 7
) -> Dict[str, Any]:
    """
    Executes full AI clinical risk intelligence analysis via Google Gemini API
    with automatic offline clinical engine fallback.
    """
    dossier = _gather_patient_clinical_dossier(db, patient_id, window_days)
    reasons = baseline_reasons or []

    api_key = get_gemini_api_key()
    if not api_key:
        logger.info("No Gemini API key available; executing offline clinical rule engine.")
        return _execute_clinical_rule_fallback(dossier, baseline_level, reasons)

    prompt = f"""You are Recovera AI Senior Clinical Intelligence System for Hospital Post-Discharge Patient Monitoring.
Analyze this comprehensive patient recovery dossier:

PATIENT DIAGNOSTIC PROFILE:
- Name: {dossier['name']} (Age: {dossier['age'] or 'N/A'}, Gender: {dossier['gender']})
- Primary Diagnosed Condition: {dossier['active_condition']} (Stage: {dossier['condition_stage']})

PREVIOUS MEDICAL HISTORY & COMORBIDITIES:
- Pre-existing Chronic Diseases: {json.dumps(dossier['medical_history']['chronic_conditions'])}
- Long-term Medications: {json.dumps(dossier['medical_history']['past_medications'])}
- Past Surgeries & Interventions: {json.dumps(dossier['medical_history']['past_surgeries'])}
- History Notes & Allergies: {dossier['medical_history']['history_notes']}

MEDICATION INTAKE TELEMETRY:
- Adherence Rate: {dossier['medication_telemetry']['adherence_pct']}%
- Total Doses Scheduled: {dossier['medication_telemetry']['total_scheduled_doses']} (Taken: {dossier['medication_telemetry']['taken_doses']}, Missed: {dossier['medication_telemetry']['missed_doses_count']})
- Active Prescriptions: {json.dumps(dossier['medication_telemetry']['active_medications'])}
- Specifically Missed Medications: {json.dumps(dossier['medication_telemetry']['missed_medication_details'])}

DAILY CHECK-IN TELEMETRY:
- Submissions Count: {dossier['symptom_telemetry']['submissions_count']}
- Maximum Pain Reported: {dossier['symptom_telemetry']['max_pain']}/10 (Recent trend: {dossier['symptom_telemetry']['recent_pains']})
- Active Fever Reported: {dossier['symptom_telemetry']['has_active_fever']}
- Recent Check-in Logs: {json.dumps(dossier['symptom_telemetry']['checkins'])}

VERBATIM CUSTOM PATIENT MESSAGES / NOTES:
{json.dumps(dossier['custom_patient_messages'], indent=2)}

BASELINE CLINICAL ENGINE EVALUATION:
- Baseline Safety Level: {baseline_level.upper()}
- Baseline Triggers: {json.dumps(reasons)}

CLINICAL EVALUATION TASK:
Synthesize the interaction between the primary condition, chronic medical history, exact medication compliance, daily telemetry, and the patient's custom messages.

Return ONLY a strict JSON object with this EXACT schema:
{{
  "ai_risk_level": "green" | "yellow" | "red",
  "clinical_score": <integer from 0 to 100>,
  "patient_facing_summary": "<1-2 empathetic, warm, plain-language sentences explaining to the patient why their status is what it is and what simple action to take>",
  "doctor_nurse_summary": {{
    "clinical_assessment": "<Detailed clinical synthesis connecting primary condition, comorbidities, and current trajectory>",
    "medication_impact": "<Analysis of missed medications, especially critical drugs like antibiotics, blood thinners, cardiac drugs>",
    "symptom_trajectory_analysis": "<Analysis of reported pain score, anatomical locations, temperature, vitals, mobility>",
    "custom_notes_sentiment_flags": [
      "<Extracted clinical warning signals, distress markers, or symptom descriptions from patient custom notes>"
    ],
    "recommended_clinical_interventions": [
      "<Concrete, prioritized clinical next steps for attending doctor and nurse>"
    ]
  }},
  "medical_history_impact": "<Sentence explaining how pre-existing conditions affect the recovery hazard>",
  "custom_message_analysis": "<Summary of patient's direct reported concerns>"
}}
"""

    candidate_models = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-flash-latest"]
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        for model_name in candidate_models:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                if resp.text:
                    parsed = json.loads(resp.text.strip())
                    parsed["model_used"] = f"Google Gemini ({model_name})"

                    # Clinical safety invariant: Rule-based RED cannot be downgraded by AI
                    if baseline_level.lower() == "red":
                        parsed["ai_risk_level"] = "red"
                        parsed["clinical_score"] = max(75, parsed.get("clinical_score", 85))

                    return parsed
            except Exception as model_err:
                logger.warning(f"Gemini model {model_name} failed: {model_err}. Trying next candidate...")
                continue

    except Exception as e:
        logger.error(f"Gemini API invocation failed: {e}")

    # Fallback to local clinical reasoning
    return _execute_clinical_rule_fallback(dossier, baseline_level, reasons)
