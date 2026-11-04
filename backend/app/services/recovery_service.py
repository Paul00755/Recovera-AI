import json
import logging
import datetime as dt
from typing import Optional, Dict, Any, List
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.models.recovery_plan import RecoveryPlan
from app.models.clinical_document import ClinicalDocument
from app.models.user import User
from app.models.patient import Patient, MedicalHistory
from app.models.medication import Medication, MedicationSchedule
from app.models.checkin import DailyCheckIn
from app.core.constants import DocStatus, Role
from app.core.audit import record_audit

logger = logging.getLogger("recovera.recovery_service")

def generate_plan_from_approved_documents(patient_id: str, db: Session, user_id: str) -> RecoveryPlan:
    """
    Aggregates discharge summaries, prescriptions, active medications, past 7 days of daily check-ins,
    and clinical history, then uses Google Gemini (or enhanced clinical synthesis) to generate
    an evidence-grounded, patient-friendly recovery care plan with dietary guidelines and activity targets.
    """
    patient = db.get(Patient, patient_id)
    if not patient or not patient.active:
        raise HTTPException(status_code=404, detail="Patient not found.")

    approved_docs = db.scalars(
        select(ClinicalDocument).where(
            ClinicalDocument.patient_id == patient_id,
            ClinicalDocument.status == DocStatus.APPROVED.value
        )
    ).all()

    # Active medications and their meal schedules
    active_meds = db.scalars(
        select(Medication).where(
            Medication.patient_id == patient_id,
            Medication.status == "active"
        )
    ).all()

    med_details = []
    med_names = []
    for m in active_meds:
        scheds = db.scalars(
            select(MedicationSchedule).where(MedicationSchedule.medication_id == m.id)
        ).all()
        sched_strs = [f"{s.label} ({s.time_hhmm}, {s.food_relation or 'after food'})" for s in scheds]
        med_names.append(m.name)
        med_details.append({
            "name": m.name,
            "dose": m.dose or "1 dose",
            "instructions": m.instructions or "",
            "meals": sched_strs or ["Daily"]
        })

    # Past 7 days of daily check-ins
    past_checkins = db.scalars(
        select(DailyCheckIn)
        .where(DailyCheckIn.patient_id == patient_id)
        .order_by(DailyCheckIn.submitted_at.desc())
        .limit(7)
    ).all()

    checkin_summaries = []
    pain_scores = []
    for c in past_checkins:
        p_val = getattr(c, "pain_level", None) if getattr(c, "pain_level", None) is not None else getattr(c, "pain_score", None)
        if p_val is not None:
            pain_scores.append(p_val)
        mobility = getattr(c, "activity_level", None) or getattr(c, "mobility_score", None) or "normal"
        checkin_summaries.append(
            f"{c.submitted_at.strftime('%Y-%m-%d')}: Pain {p_val if p_val is not None else 'N/A'}/10, Mobility: {mobility}, Fever: {getattr(c, 'fever', False)}, Notes: {getattr(c, 'notes', 'None')}"
        )

    avg_pain = round(sum(pain_scores) / len(pain_scores), 1) if pain_scores else None

    # Medical history
    med_hist = db.scalar(select(MedicalHistory).where(MedicalHistory.patient_id == patient_id))
    conditions = []
    allergies = []
    if med_hist:
        if med_hist.conditions:
            conditions.extend(med_hist.conditions if isinstance(med_hist.conditions, list) else [med_hist.conditions])
        if med_hist.notes:
            allergies.append(med_hist.notes)

    # Document instructions & warnings
    source_versions = {}
    extracted_instructions = []
    for doc in approved_docs:
        source_versions[doc.id] = {
            "version": doc.version,
            "filename": doc.filename,
            "doc_type": doc.doc_type
        }
        struct = doc.structured_data or {}
        if doc.doc_type == "prescription":
            for med in struct.get("medications", []):
                m_name = med.get("medicine_name") or med.get("name")
                if m_name and m_name not in med_names:
                    med_names.append(m_name)
        elif doc.doc_type in ("discharge_summary", "recovery_instructions"):
            instr = struct.get("instructions")
            if instr:
                if isinstance(instr, str):
                    for line in instr.split("\n"):
                        s = line.strip(" -*")
                        if len(s) > 10:
                            extracted_instructions.append(s)
                elif isinstance(instr, list):
                    extracted_instructions.extend([str(x).strip(" -*") for x in instr if len(str(x).strip()) > 10])

    if not approved_docs and not active_meds and not checkin_summaries and not conditions:
        raise HTTPException(
            status_code=400,
            detail="Cannot generate recovery plan: No approved documents, active medications, or check-in history found for this patient."
        )

    # Call Gemini to synthesize an optimized plan
    generated_plan = None
    model_used = "Recovera Clinical Synthesis Engine"

    cond_str = patient.active_condition or "General Recovery"
    stage_str = patient.condition_stage or "In Recovery"
    notes_str = patient.condition_notes or "None"

    try:
        from app.services.recovery_ai_service import get_gemini_api_key
        api_key = get_gemini_api_key()
        if api_key:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            prompt = f"""
You are Recovera AI, an expert clinical recovery intelligence engine.
Synthesize a personalized, simple, and patient-friendly recovery care plan for the following patient.

PATIENT CONTEXT:
- Name: {patient.name or patient.patient_code}
- Active Condition / Diagnosis: {cond_str} (Stage: {stage_str})
- Clinical Condition Notes: {notes_str}
- Active Prescriptions ({len(med_details)}): {json.dumps(med_details)}
- Past 7 Days Symptom Check-ins: {json.dumps(checkin_summaries)}
- Average Pain Score: {avg_pain if avg_pain is not None else 'Not reported'}/10
- Known Conditions: {json.dumps(conditions)}
- Allergies / Notes: {json.dumps(allergies)}
- Clinical Document Instructions: {json.dumps(extracted_instructions[:5])}

CLINICAL & LIFESTYLE REQUIREMENTS:
1. Address the patient's active condition directly ({cond_str}, {stage_str}).
2. Use clear, everyday, and easily understandable language with practical bullet points.
3. Provide actionable dietary and hydration advice:
   - Hydration: drink warm or boiled water regularly throughout the day.
   - Energy restoration: explain that fever/illness depletes body energy and can cause dizziness. Recommend easily digestible, nutrient-dense foods (e.g., warm milk, bread, porridge, soft rice, clear broths, fresh fruits).
   - Foods to avoid: explicitly warn against heavy, hard-to-digest meats (chicken, mutton, beef) and greasy/spicy foods during acute illness.
4. Provide daily activity and rest guidance:
   - Sleep: prioritize plenty of restorative sleep and go to bed before 10 PM.
   - Activity balance: do not workout or over-exert, but also do not lie in bed all day without movement—keep physical activity gentle and moderate (gentle indoor walking).
5. Provide clear guidance on treatment adherence:
   - Take all medicines punctually as scheduled around Breakfast, Lunch, and Dinner.
   - Record every dose in the medication intake tracker and complete the daily check-in to keep doctors informed.
   - Advise patient to consult the doctor immediately if experiencing severe pain, high fever, or abnormal symptoms.
6. List 4-5 key warning signs requiring immediate clinical contact.
7. Provide high-quality Tamil ('ta') and Hindi ('hi') translations for all sections so the patient can switch languages.

RESPOND IN VALID JSON ONLY matching this exact structure:
{{
  "general": "Simple recovery overview statement personalized to their condition.",
  "diet": "Specific food and hydration guidelines (warm water, digestible foods like bread/milk, avoiding heavy meats like chicken/mutton/beef).",
  "activity": "Rest before 10 PM, avoiding heavy workout while keeping movement moderate (not lying in bed all day).",
  "wound_care": "Simple wound, incision, or personal hygiene guidelines.",
  "medications_summary": "Friendly guide to taking their morning, afternoon, and night medicines and logging them.",
  "warning_signs": [
    "High fever above 38°C (100.4°F)",
    "Severe or suddenly worsening pain",
    "Expanding redness or warmth at wound site",
    "Shortness of breath or severe dizziness"
  ],
  "translations": {{
    "ta": {{
      "general": "...",
      "diet": "...",
      "activity": "...",
      "wound_care": "...",
      "medications_summary": "...",
      "warning_signs": ["...", "..."]
    }},
    "hi": {{
      "general": "...",
      "diet": "...",
      "activity": "...",
      "wound_care": "...",
      "medications_summary": "...",
      "warning_signs": ["...", "..."]
    }}
  }}
}}
"""
            for model_name in ["gemini-3.5-flash-lite", "gemini-flash-latest", "gemini-2.5-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            response_mime_type="application/json"
                        )
                    )
                    if response and response.text:
                        raw = response.text.strip()
                        # Clean markdown fence if present
                        if raw.startswith("```json"):
                            raw = raw[7:]
                        if raw.startswith("```"):
                            raw = raw[3:]
                        if raw.endswith("```"):
                            raw = raw[:-3]
                        parsed = json.loads(raw.strip())
                        if "diet" in parsed and "activity" in parsed:
                            generated_plan = parsed
                            model_used = f"Google {model_name}"
                            break
                except Exception as model_err:
                    logger.warning("Gemini model %s generation failed: %s", model_name, model_err)
    except Exception as e:
        logger.warning("Gemini recovery plan generation encountered exception: %s", e)

    # Deterministic Clinical Synthesis Fallback if Gemini unavailable
    if not generated_plan:
        is_fever_case = any(
            "fever" in str(x).lower()
            for x in [patient.active_condition or "", notes_str, checkin_summaries]
        )

        med_summary_text = (
            f"Take your active prescribed medications ({', '.join(med_names)}) strictly according to their Breakfast, Lunch, and Dinner schedules. Log each dose in the medication tracker and complete your daily check-in to keep your recovery team informed."
            if med_names else "Take all recovery medications punctually according to your meal schedule, log them in your tracker, and submit your daily check-in."
        )

        if is_fever_case:
            activity_text = (
                "Take plenty of restful sleep and go to bed before 10 PM. Fever reduces your body energy and can make you dizzy, so avoid strenuous workouts or heavy exercise. However, do not lie motionless in bed all day—keep physical activity gentle and moderate, such as short 5-10 minute indoor walks."
            )
            diet_text = (
                "Drink warm or boiled water regularly throughout the day to stay hydrated. Since fever depletes your energy and causes dizziness, eat nutrient-dense and easily digestible foods like warm milk, bread, porridge, soft rice, and steamed vegetables. Strictly avoid heavy, non-digestible foods like chicken, mutton, or beef, and avoid oily/spicy dishes during acute recovery."
            )
            general_text = (
                f"Personalized Recovery Guide for {patient.active_condition or 'High Fever'}: Prioritize hydration with warm water, restorative sleep before 10 PM, easily digestible nutrition, and prompt communication with your doctor if symptoms worsen."
            )
        else:
            activity_text = (
                "Prioritize restful sleep and retire before 10 PM. Engage in gentle 10-15 minute indoor walks 2-3 times daily to keep blood circulating. Avoid strenuous workouts or heavy lifting (>5 kg), but avoid staying in bed all day—keep body activity moderate as tolerated."
                if (avg_pain is None or avg_pain < 4) else
                "Prioritize bed rest, sleep before 10 PM, and gentle supported movement. Take short 5-minute walks as tolerated. Avoid heavy workouts, bending, or straining until pain and inflammation subside."
            )
            diet_text = (
                "Drink plenty of warm or clean water (8-10 glasses daily). Eat light, nutrient-rich meals with easily digestible foods (warm milk, bread, vegetable broths, soft grains). Avoid heavy, non-digestible meats like chicken, mutton, or beef, as well as excessively oily or spicy foods that strain digestion while on medications."
            )
            general_text = (
                f"Personalized Recovery Guide: Follow your clinical instructions, rest adequately, maintain light nutritious meals, and log your daily progress in the patient portal."
            )

        warning_signs = [
            "High fever (>38°C / 100.4°F) or fever that does not decrease after medication",
            "Severe, sharp, or suddenly worsening pain not controlled by prescribed drugs",
            "Shortness of breath, persistent chest tightness, severe dizziness, or fainting",
            "Inability to keep liquids down, intractable vomiting, or severe dehydration",
            "Spreading redness, warmth, foul-smelling discharge, or wound breakdown"
        ]

        generated_plan = {
            "general": general_text,
            "diet": diet_text,
            "activity": activity_text,
            "wound_care": "Keep dressing clean and dry. Wash hands thoroughly before touching any wound or incision area. Inspect daily for unusual redness or swelling.",
            "medications_summary": med_summary_text,
            "warning_signs": warning_signs,
            "translations": {
                "ta": {
                    "general": f"உங்கள் மீட்சிக்கான வழிகாட்டுதல்: வெதுவெதுப்பான நீரைக் குடிக்கவும், இரவு 10 மணிக்குள் தூங்கவும், எளிதில் ஜீரணமாகும் உணவுகளை உட்கொள்ளவும்.",
                    "diet": "சூடான அல்லது காய்ச்சிய நீரைக் குடிக்கவும். காய்ச்சல் உடலின் ஆற்றலைக் குறைத்து தலைசுற்றலை ஏற்படுத்தும் என்பதால், ரொட்டி, பால், கஞ்சி போன்ற எளிதில் ஜீரணமாகும் சத்தான உணவுகளை உண்ணுங்கள். கோழி, ஆடு, மாடு போன்ற அசைவ உணவுகளையும் காரமான உணவுகளையும் தவிர்க்கவும்.",
                    "activity": "நன்றாக ஓய்வெடுத்து இரவு 10 மணிக்குள் தூங்கவும். அதிக உடற்பயிற்சி செய்ய வேண்டாம், அதே நேரம் நாள் முழுவதும் படுக்கையிலேயே கிடக்காமல் மிதமான நடைப்பயிற்சி மேற்கொள்ளவும்.",
                    "wound_care": "காயத்தைச் சுத்தமாகவும் உலர்வாகவும் வைத்திருக்கவும். தொடுவதற்கு முன் கைகளை நன்றாகக் கழுவவும்.",
                    "medications_summary": f"உங்கள் மருந்துகளை ({', '.join(med_names)}) காலை, மதிய, இரவு நேரங்களில் தவறாமல் உட்கொண்டு, மருந்து டிராக்கரில் பதிவு செய்யவும்.",
                    "warning_signs": [
                        "100.4°F-க்கு மேல் கடுமையான காய்ச்சல்",
                        "மருந்துகளால் குறையாத கடுமையான வலி",
                        "மூச்சுத் திணறல் அல்லது கடுமையான தலைசுற்றல்",
                        "காயத்தைச் சுற்றி சிவத்தல் அல்லது சீழ் வடிதல்"
                    ]
                },
                "hi": {
                    "general": "आपकी रिकवरी के लिए निर्देश: गुनगुना पानी पिएं, रात 10 बजे से पहले सोएं, और सुपाच्य पौष्टिक आहार लें।",
                    "diet": "दिनभर गुनगुना या उबला हुआ पानी पिएं। बुखार शरीर की ऊर्जा कम कर चक्कर ला सकता है, इसलिए ब्रेड, दूध, दलिया जैसे आसानी से पचने वाले पौष्टिक भोजन लें। चिकन, मटन, बीफ जैसे भारी मांसाहार और तैलीय भोजन से बचें।",
                    "activity": "भरपूर आराम करें और रात 10 बजे से पहले सोएं। बहुत भारी व्यायाम न करें, साथ ही दिनभर बिस्तर पर न पड़े रहें—हल्की सैर करें और शरीर को मध्यम रूप से सक्रिय रखें।",
                    "wound_care": "घाव को साफ और सूखा रखें। पट्टी छूने से पहले हाथों को अच्छी तरह धोएं।",
                    "medications_summary": f"अपनी निर्धारित दवाओं ({', '.join(med_names)}) को सुबह, दोपहर और रात के भोजन के अनुसार समय पर लें और ट्रैकर में दर्ज करें।",
                    "warning_signs": [
                        "100.4°F से अधिक तेज बुखार",
                        "दवाओं से कम न होने वाला गंभीर दर्द",
                        "सांस लेने में कठिनाई या गंभीर चक्कर आना",
                        "घाव के आसपास बढ़ती लालिमा या रिसाव"
                    ]
                }
            }
        }

    instructions = {
        "general": generated_plan.get("general", ""),
        "diet": generated_plan.get("diet", ""),
        "activity": generated_plan.get("activity", ""),
        "wound_care": generated_plan.get("wound_care", ""),
        "medications_summary": generated_plan.get("medications_summary", "")
    }
    if extracted_instructions:
        instructions["hospital_specific"] = extracted_instructions[:5]

    warning_signs = generated_plan.get("warning_signs", [])
    translations = generated_plan.get("translations", {})

    # Find existing plan or create new
    existing_plan = db.scalar(
        select(RecoveryPlan).where(RecoveryPlan.patient_id == patient_id).order_by(RecoveryPlan.created_at.desc())
    )

    if existing_plan:
        plan = existing_plan
        plan.version = (plan.version or 1) + 1
        plan.status = DocStatus.DRAFT.value
        plan.source_versions = source_versions
        plan.instructions = instructions
        plan.warning_signs = warning_signs
        plan.content = {
            "instructions": instructions,
            "warning_signs": warning_signs,
            "translations": translations,
            "generated_with": model_used,
            "version": plan.version,
            "patient_context": {
                "active_med_count": len(active_meds),
                "checkin_count": len(checkin_summaries),
                "average_pain": avg_pain
            }
        }
        plan.updated_at = dt.datetime.utcnow()
    else:
        plan = RecoveryPlan(
            patient_id=patient_id,
            version=1,
            status=DocStatus.DRAFT.value,
            source_versions=source_versions,
            instructions=instructions,
            warning_signs=warning_signs,
            content={
                "instructions": instructions,
                "warning_signs": warning_signs,
                "translations": translations,
                "generated_with": model_used,
                "version": 1,
                "patient_context": {
                    "active_med_count": len(active_meds),
                    "checkin_count": len(checkin_summaries),
                    "average_pain": avg_pain
                }
            },
            created_at=dt.datetime.utcnow(),
            updated_at=dt.datetime.utcnow()
        )
        db.add(plan)

    db.flush()

    record_audit(
        db=db,
        actor_id=user_id,
        action="recovery_plan_generated",
        target_type="recovery_plan",
        target_id=plan.id,
        new_values={"version": plan.version, "status": plan.status, "model_used": model_used}
    )

    # Suggestion 4: Alert patient of updated recovery plan
    try:
        from app.services.notification_service import create_in_app_notification
        if patient.user_id:
            create_in_app_notification(
                db=db,
                user_id=patient.user_id,
                title="Recovery Plan Ready",
                message=f"Your personalized clinical recovery plan (v{plan.version}) has been generated and updated by your clinical care team.",
                category="recovery_plan",
                link="#/patient/recovery"
            )
    except Exception as e:
        logger.warning("Failed to dispatch recovery plan notification: %s", e)

    db.commit()
    db.refresh(plan)
    return plan

def invalidate_plan_on_source_change(patient_id: str, db: Session, document_id: str) -> Optional[RecoveryPlan]:
    """
    When an approved clinical document is modified, rejected, or superseded,
    the active published recovery plan is transitioned to NEEDS_REVIEW
    to prevent patients from viewing outdated or invalidated care guidelines.
    """
    plan = db.scalar(
        select(RecoveryPlan).where(RecoveryPlan.patient_id == patient_id).order_by(RecoveryPlan.created_at.desc())
    )
    if not plan:
        return None

    # Check if document was part of source or plan was published
    source_vers = plan.source_versions or {}
    if document_id in source_vers or plan.status == DocStatus.PUBLISHED.value:
        old_status = plan.status
        plan.status = DocStatus.NEEDS_REVIEW.value
        plan.updated_at = dt.datetime.utcnow()
        
        record_audit(
            db=db,
            actor_id=None,
            action="recovery_plan_invalidated_source_changed",
            target_type="recovery_plan",
            target_id=plan.id,
            old_values={"status": old_status},
            new_values={"status": DocStatus.NEEDS_REVIEW.value, "trigger_document_id": document_id}
        )
        db.commit()
        return plan

    return None

def publish_recovery_plan(
    patient_id: str,
    db: Session,
    user: User,
    instructions: Optional[Dict[str, Any]] = None,
    warning_signs: Optional[List[Any]] = None
) -> RecoveryPlan:
    """
    Doctor or Admin reviews and officially publishes a recovery plan.
    Nurses cannot publish recovery plans (enforcing RBAC clinical responsibility).
    """
    if user.role not in (Role.DOCTOR.value, Role.ADMIN.value):
        raise HTTPException(
            status_code=403,
            detail="Forbidden: Only Doctors and Administrators are authorized to publish recovery plans."
        )

    plan = db.scalar(
        select(RecoveryPlan).where(RecoveryPlan.patient_id == patient_id).order_by(RecoveryPlan.created_at.desc())
    )
    if not plan:
        raise HTTPException(status_code=404, detail="No recovery plan found for this patient.")

    if instructions is not None:
        plan.instructions = instructions
    if warning_signs is not None:
        plan.warning_signs = warning_signs

    plan.status = DocStatus.PUBLISHED.value
    plan.approved_by = user.id
    plan.approved_at = dt.datetime.utcnow()
    plan.updated_at = dt.datetime.utcnow()
    plan.content = {
        "instructions": plan.instructions,
        "warning_signs": plan.warning_signs,
        "version": plan.version
    }

    record_audit(
        db=db,
        actor_id=user.id,
        action="recovery_plan_published",
        target_type="recovery_plan",
        target_id=plan.id,
        new_values={"status": plan.status, "version": plan.version}
    )

    # Suggestion 4: Notify patient of published recovery plan guidelines
    try:
        from app.models.patient import Patient
        from app.services.notification_service import notify_patient_recovery_plan_updated
        patient_rec = db.get(Patient, patient_id)
        if patient_rec:
            notify_patient_recovery_plan_updated(db=db, patient=patient_rec, doctor=user)
    except Exception as e:
        import logging
        logging.getLogger("recovera.recovery_plan").warning("Failed to notify patient of published plan: %s", e)

    db.commit()
    db.refresh(plan)
    return plan
