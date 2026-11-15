"""
Recovera AI – Advanced NLP Medical Entity Extraction & Normalization Engine
Extracts, validates, and normalizes structured clinical information from raw OCR text:
- Patient details (name, age, gender, patient code)
- Medicine names, generic equivalents, and strengths
- Dosages, routes, and frequencies
- Dosing schedules (1-0-1, 1-0-0, 0-0-1, 1-1-1, etc.)
- Food relations (before_food, after_food, with_food)
- Treatment durations (days, weeks, months)
- Clinical diagnosis & primary symptoms
- Clinical instructions & precautions
- Follow-up dates and scheduling
- Uncertainty detection: flags uncertain fields for human verification
"""

import re
import datetime as dt
from typing import Dict, Any, List, Optional, Tuple


class NLPMedicalExtractor:
    """
    Rule-guided and dictionary-grounded NLP medical entity extractor.
    Performs entity extraction, canonical normalization, and clinical constraint validation.
    """

    KNOWN_DRUGS_KB = [
        {"name": "Paracetamol", "generic": "Acetaminophen", "category": "Analgesic / Antipyretic", "std_strengths": ["500mg", "650mg", "1000mg"]},
        {"name": "Pan 40", "generic": "Pantoprazole Sodium", "category": "Proton Pump Inhibitor", "std_strengths": ["40mg", "20mg"]},
        {"name": "Pantoprazole", "generic": "Pantoprazole Sodium", "category": "Proton Pump Inhibitor", "std_strengths": ["40mg", "20mg"]},
        {"name": "Acedo MR", "generic": "Aceclofenac + Paracetamol + Chlorzoxazone", "category": "Muscle Relaxant", "std_strengths": ["100mg/325mg/250mg"]},
        {"name": "Deflazacort", "generic": "Deflazacort", "category": "Corticosteroid", "std_strengths": ["6mg", "12mg", "30mg"]},
        {"name": "Tolperisone", "generic": "Tolperisone Hydrochloride", "category": "Centrally Acting Muscle Relaxant", "std_strengths": ["150mg", "50mg"]},
        {"name": "Amoxicillin", "generic": "Amoxicillin Trihydrate", "category": "Beta-Lactam Antibiotic", "std_strengths": ["250mg", "500mg", "875mg"]},
        {"name": "Azithromycin", "generic": "Azithromycin Dihydrate", "category": "Macrolide Antibiotic", "std_strengths": ["250mg", "500mg"]},
        {"name": "Cefixime", "generic": "Cefixime", "category": "Cephalosporin Antibiotic", "std_strengths": ["100mg", "200mg", "400mg"]},
        {"name": "Ciprofloxacin", "generic": "Ciprofloxacin HCl", "category": "Fluoroquinolone Antibiotic", "std_strengths": ["250mg", "500mg"]},
        {"name": "Metformin", "generic": "Metformin Hydrochloride", "category": "Biguanide Antidiabetic", "std_strengths": ["500mg", "850mg", "1000mg"]},
        {"name": "Atorvastatin", "generic": "Atorvastatin Calcium", "category": "Statin / Lipid-lowering", "std_strengths": ["10mg", "20mg", "40mg", "80mg"]},
        {"name": "Losartan", "generic": "Losartan Potassium", "category": "Angiotensin II Receptor Blocker", "std_strengths": ["25mg", "50mg", "100mg"]},
        {"name": "Amlodipine", "generic": "Amlodipine Besylate", "category": "Calcium Channel Blocker", "std_strengths": ["2.5mg", "5mg", "10mg"]},
        {"name": "Metoprolol", "generic": "Metoprolol Succinate", "category": "Beta Blocker", "std_strengths": ["25mg", "50mg", "100mg"]},
        {"name": "Cetirizine", "generic": "Cetirizine Hydrochloride", "category": "Second-generation Antihistamine", "std_strengths": ["5mg", "10mg"]},
        {"name": "Montelukast", "generic": "Montelukast Sodium", "category": "Leukotriene Receptor Antagonist", "std_strengths": ["4mg", "5mg", "10mg"]},
        {"name": "Ibuprofen", "generic": "Ibuprofen", "category": "NSAID", "std_strengths": ["200mg", "400mg", "600mg"]},
        {"name": "Tramadol", "generic": "Tramadol Hydrochloride", "category": "Opioid Analgesic", "std_strengths": ["50mg", "100mg"]},
        {"name": "Doxycycline", "generic": "Doxycycline Hyclate", "category": "Tetracycline Antibiotic", "std_strengths": ["100mg"]},
        {"name": "Lisinopril", "generic": "Lisinopril", "category": "ACE Inhibitor", "std_strengths": ["5mg", "10mg", "20mg"]},
        {"name": "Omeprazole", "generic": "Omeprazole", "category": "Proton Pump Inhibitor", "std_strengths": ["20mg", "40mg"]},
    ]

    SCHEDULE_NOTATIONS = {
        "1-0-0": {"label": "Morning only", "frequency": "once_daily", "daily_times": 1, "schedules": [{"label": "Morning", "time_hhmm": "08:00"}]},
        "0-1-0": {"label": "Afternoon only", "frequency": "once_daily", "daily_times": 1, "schedules": [{"label": "Afternoon", "time_hhmm": "13:00"}]},
        "0-0-1": {"label": "Night only", "frequency": "once_daily", "daily_times": 1, "schedules": [{"label": "Night", "time_hhmm": "20:00"}]},
        "1-0-1": {"label": "Morning & Night", "frequency": "twice_daily", "daily_times": 2, "schedules": [{"label": "Morning", "time_hhmm": "08:00"}, {"label": "Night", "time_hhmm": "20:00"}]},
        "1-1-0": {"label": "Morning & Afternoon", "frequency": "twice_daily", "daily_times": 2, "schedules": [{"label": "Morning", "time_hhmm": "08:00"}, {"label": "Afternoon", "time_hhmm": "13:00"}]},
        "1-1-1": {"label": "Morning, Afternoon & Night", "frequency": "thrice_daily", "daily_times": 3, "schedules": [{"label": "Morning", "time_hhmm": "08:00"}, {"label": "Afternoon", "time_hhmm": "13:00"}, {"label": "Night", "time_hhmm": "20:00"}]},
        "1-1-1-1": {"label": "Four times daily", "frequency": "four_times_daily", "daily_times": 4, "schedules": [{"label": "Morning", "time_hhmm": "08:00"}, {"label": "Noon", "time_hhmm": "12:00"}, {"label": "Evening", "time_hhmm": "16:00"}, {"label": "Night", "time_hhmm": "20:00"}]},
        "0-0-0-1": {"label": "Bedtime only", "frequency": "once_daily", "daily_times": 1, "schedules": [{"label": "Bedtime", "time_hhmm": "21:30"}]},
    }

    @classmethod
    def extract_structured_clinical_entities(cls, raw_ocr_text: str) -> Dict[str, Any]:
        """
        Main NLP pipeline parsing unstructured raw OCR into verified medical entities.
        """
        text = raw_ocr_text or ""
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        patient_info = cls._extract_patient_details(text)
        diagnosis_info = cls._extract_diagnosis_and_symptoms(text)
        medications = cls._extract_medications(lines)
        instructions = cls._extract_clinical_instructions(text)
        follow_up = cls._extract_follow_up(text)

        total_extracted_fields = (
            len(medications) * 6 +
            (1 if patient_info.get("name") else 0) +
            (1 if diagnosis_info.get("diagnosis") else 0) +
            (1 if follow_up.get("follow_up_date") else 0)
        )
        needs_verification_count = sum(1 for m in medications if m.get("needs_verification"))

        overall_confidence = round(max(0.60, min(0.98, 0.95 - (needs_verification_count * 0.08))), 2)

        return {
            "patient": patient_info,
            "diagnosis": diagnosis_info.get("diagnosis"),
            "symptoms": diagnosis_info.get("symptoms", []),
            "medications": medications,
            "clinical_instructions": instructions,
            "follow_up": follow_up,
            "extraction_confidence": overall_confidence,
            "requires_human_verification": bool(needs_verification_count > 0 or not medications),
            "verification_reasons": [
                f"Medication '{m['name']}' requires manual dosage confirmation"
                for m in medications if m.get("needs_verification")
            ]
        }

    @classmethod
    def _extract_patient_details(cls, text: str) -> Dict[str, Any]:
        details = {"name": None, "age": None, "gender": None, "patient_code": None}

        # Patient Name patterns
        m_name = re.search(r"(?i)(?:patient(?:\s+name)?|pt(?:\s+name)?|mr\.|mrs\.|ms\.)[:\s]+([a-zA-Z\s\.]{2,35})", text)
        if m_name:
            cand = m_name.group(1).strip()
            # Clean common trailing labels
            cand = re.sub(r"(?i)\b(age|sex|gender|date|dr|rx|diagnosis)\b.*$", "", cand).strip()
            if len(cand) >= 2:
                details["name"] = cand.title()

        # Age patterns: "Age: 45", "45 Y", "45 yrs", "45/M"
        m_age = re.search(r"(?i)(?:age[:\s]+)?(\d{1,3})\s*(?:y(?:ears?|rs?)?|yo\b)", text)
        if not m_age:
            m_age = re.search(r"\b(\d{1,3})\s*(?:/|\s+)?(?:M|F|Male|Female)\b", text, re.IGNORECASE)
        if m_age:
            try:
                age_val = int(m_age.group(1))
                if 1 <= age_val <= 120:
                    details["age"] = age_val
            except Exception:
                pass

        # Gender patterns
        if re.search(r"(?i)\b(female|woman|\bF\b)", text):
            details["gender"] = "Female"
        elif re.search(r"(?i)\b(male|man|\bM\b)", text):
            details["gender"] = "Male"

        # Patient Code: "PAT-1029", "P-4820", "ID: 29482"
        m_code = re.search(r"(?i)(?:patient\s*(?:id|code|no)|id|mrn)[:\s]+([A-Z0-9\-]+)", text)
        if m_code:
            details["patient_code"] = m_code.group(1).strip().upper()

        return details

    @classmethod
    def _extract_diagnosis_and_symptoms(cls, text: str) -> Dict[str, Any]:
        res = {"diagnosis": None, "symptoms": []}

        # Diagnosis extraction
        m_diag = re.search(r"(?i)(?:diagnosis|dx|impression|condition|assessment)[:\s]+([^\n\.;]+)", text)
        if m_diag:
            cand = m_diag.group(1).strip()
            cand = re.sub(r"(?i)\b(rx|medications|plan|advise|instructions)\b.*$", "", cand).strip()
            if len(cand) >= 3:
                res["diagnosis"] = cand.title()

        # Symptom detection
        symptom_lexicon = [
            "fever", "headache", "body ache", "joint pain", "nausea", "vomiting",
            "cough", "cold", "sore throat", "shortness of breath", "abdominal pain",
            "chest discomfort", "dizziness", "weakness", "fatigue", "wound pain",
            "swelling", "edema", "burning micturition", "rash", "loose stools"
        ]
        lower_t = text.lower()
        found_sym = [s.title() for s in symptom_lexicon if re.search(rf"\b{re.escape(s)}\b", lower_t)]
        res["symptoms"] = list(dict.fromkeys(found_sym))

        return res

    @classmethod
    def _extract_medications(cls, lines: List[str]) -> List[Dict[str, Any]]:
        meds = []
        seen_names = set()

        for line in lines:
            if not line or len(line) < 3:
                continue

            # Detect candidate medicine name
            matched_kb_drug = None
            for kb_entry in cls.KNOWN_DRUGS_KB:
                kb_name = kb_entry["name"]
                if re.search(rf"\b{re.escape(kb_name)}\b", line, re.IGNORECASE):
                    matched_kb_drug = kb_entry
                    break

            med_name = None
            generic_name = None
            if matched_kb_drug:
                med_name = matched_kb_drug["name"]
                generic_name = matched_kb_drug["generic"]
            else:
                # Regex heuristic for medicine names: e.g. "Tab Paracetamol", "Cap Amoxy", "T. Deflazacort"
                m_drug = re.search(r"(?i)\b(?:t\.|tab(?:let)?|cap(?:sule)?|syr(?:up)?|inj(?:ection)?)\s+([a-zA-Z0-9\-\s]{3,25})", line)
                if m_drug:
                    cand = m_drug.group(1).strip()
                    # Strip trailing schedule acronyms
                    cand = re.sub(r"(?i)\s+(?:1\s*[-/\.]?\s*)?(?:od|bd|bid|tds|tid|hs|sos|[01][\s\.\-_]+[01][\s\.\-_]+[01]).*$", "", cand).strip()
                    if cand and len(cand) >= 3 and not re.search(r"(?i)^(patient|doctor|date|clinic|hospital)", cand):
                        med_name = cand.title()

            if not med_name:
                continue

            norm_key = med_name.lower()
            if norm_key in seen_names:
                continue
            seen_names.add(norm_key)

            # Strength extraction: e.g. "650mg", "40 mg", "500 mg", "6 mg"
            m_strength = re.search(r"\b(\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu))\b", line, re.IGNORECASE)
            strength_val = m_strength.group(1).replace(" ", "").lower() if m_strength else None
            if not strength_val and matched_kb_drug and matched_kb_drug["std_strengths"]:
                # Default to primary standard strength
                strength_val = matched_kb_drug["std_strengths"][0]

            # Dosage form / count: "1 tab", "1 capsule", "2 tabs", "5 ml"
            m_dose = re.search(r"\b(\d+)\s*(?:tab(?:let)?|cap(?:sule)?|pill|tsp|ml|puff)s?\b", line, re.IGNORECASE)
            dose_val = f"{m_dose.group(1)} tablet" if m_dose else "1 tablet"

            # Period & Schedule parsing
            schedule_info = cls._parse_schedule(line)

            # Food relation
            food_rel = "after_food"
            if re.search(r"(?i)before\s+food|pre\s*meal|empty\s*stomach|a\.c\.", line):
                food_rel = "before_food"
            elif re.search(r"(?i)with\s+food|with\s+meals?", line):
                food_rel = "with_food"

            # Treatment Duration in days
            duration_days, duration_label = cls._parse_duration(line)

            # Verification flag: if strength wasn't detected in line and drug has variable strengths
            needs_verification = bool(not m_strength and (not matched_kb_drug or len(matched_kb_drug["std_strengths"]) > 1))

            meds.append({
                "name": med_name,
                "generic_name": generic_name or med_name,
                "strength": strength_val or "Unspecified",
                "dose": dose_val,
                "dosage": strength_val or dose_val,
                "period": schedule_info["notation"],
                "frequency": schedule_info["frequency"],
                "frequency_label": schedule_info["label"],
                "daily_times": schedule_info["daily_times"],
                "schedules": schedule_info["schedules"],
                "food_relation": food_rel,
                "duration_days": duration_days,
                "duration_label": duration_label,
                "route": "oral",
                "needs_verification": needs_verification,
                "raw_source_line": line.strip()
            })

        return meds

    @classmethod
    def _parse_schedule(cls, text: str) -> Dict[str, Any]:
        """Parses schedule notation: 1-0-1, 1 0 0, 1.0.0, 1-1-1, or clinical acronyms."""
        # 1. 1s and 0s pattern (flexible separators)
        m_code = re.search(r"\b([01oOqQdDlI|!])[\s\.\-\/_]+([01oOqQdDlI|!])[\s\.\-\/_]+([01oOqQdDlI|!])(?:[\s\.\-\/_]+([01oOqQdDlI|!]))?\b", text)
        if m_code:
            def norm_c(c):
                return "0" if c in "0oOqQdD" else "1"
            parts = [norm_c(m_code.group(1)), norm_c(m_code.group(2)), norm_c(m_code.group(3))]
            if m_code.group(4):
                parts.append(norm_c(m_code.group(4)))
            code = "-".join(parts)
            if code in cls.SCHEDULE_NOTATIONS:
                res = cls.SCHEDULE_NOTATIONS[code].copy()
                res["notation"] = code
                return res

        # 2. Latin acronyms
        lower = text.lower()
        if re.search(r"(?i)\b(?:1\s*[-/\.]?\s*)?(?:od|once\s+daily|every\s+morning)\b", lower):
            res = cls.SCHEDULE_NOTATIONS["1-0-0"].copy()
            res["notation"] = "1-0-0"
            return res
        if re.search(r"(?i)\b(?:1\s*[-/\.]?\s*)?(?:bd|bid|twice\s+daily|every\s+12\s*h)\b", lower):
            res = cls.SCHEDULE_NOTATIONS["1-0-1"].copy()
            res["notation"] = "1-0-1"
            return res
        if re.search(r"(?i)\b(?:1\s*[-/\.]?\s*)?(?:tds|tid|thrice\s+daily|three\s+times)\b", lower):
            res = cls.SCHEDULE_NOTATIONS["1-1-1"].copy()
            res["notation"] = "1-1-1"
            return res
        if re.search(r"(?i)\b(?:1\s*[-/\.]?\s*)?(?:hs|qhs|bedtime|night\s+only)\b", lower):
            res = cls.SCHEDULE_NOTATIONS["0-0-1"].copy()
            res["notation"] = "0-0-1"
            return res
        if re.search(r"(?i)\b(?:1\s*[-/\.]?\s*)?(?:qid|four\s+times)\b", lower):
            res = cls.SCHEDULE_NOTATIONS["1-1-1-1"].copy()
            res["notation"] = "1-1-1-1"
            return res

        # Default fallback
        res = cls.SCHEDULE_NOTATIONS["1-0-1"].copy()
        res["notation"] = "1-0-1"
        return res

    @classmethod
    def _parse_duration(cls, text: str) -> Tuple[int, str]:
        """Parses duration in days: e.g. 5 days, 1 week, 14d, 1 month."""
        # Weeks
        m_wk = re.search(r"(?i)\b(\d+)\s*(?:week|wk)s?\b", text)
        if m_wk:
            wks = int(m_wk.group(1))
            days = wks * 7
            return days, f"{days} days ({wks} wk)"

        # Months
        m_mo = re.search(r"(?i)\b(\d+)\s*(?:month|mo)s?\b", text)
        if m_mo:
            mos = int(m_mo.group(1))
            days = mos * 30
            return days, f"{days} days ({mos} mo)"

        # Days
        m_day = re.search(r"(?i)(?:for|x)?\s*(\d{1,3})\s*(?:days?|d\b)", text)
        if m_day:
            days = int(m_day.group(1))
            if 1 <= days <= 365:
                return days, f"{days} days"

        # Trailing multiplier e.g. "x 5"
        m_x = re.search(r"(?i)[xX/]\s*(\d{1,2})\b", text)
        if m_x:
            days = int(m_x.group(1))
            if 1 <= days <= 90:
                return days, f"{days} days"

        return 5, "5 days"

    @classmethod
    def _extract_clinical_instructions(cls, text: str) -> List[str]:
        instructions = []
        key_phrases = [
            r"(?i)(?:instructions?|advice|notes?)[:\s]+([^\n]+)",
            r"(?i)(drink\s+plenty\s+of\s+water[^\n\.]*)",
            r"(?i)(avoid\s+heavy\s+lifting[^\n\.]*)",
            r"(?i)(complete\s+full\s+antibiotic\s+course[^\n\.]*)",
            r"(?i)(take\s+with\s+warm\s+water[^\n\.]*)",
            r"(?i)(maintain\s+wound\s+cleanliness[^\n\.]*)",
            r"(?i)(monitor\s+body\s+temperature[^\n\.]*)",
        ]
        for pat in key_phrases:
            for match in re.finditer(pat, text):
                cand = match.group(1).strip()
                if cand and len(cand) >= 5 and cand not in instructions:
                    instructions.append(cand.capitalize())
        return instructions

    @classmethod
    def _extract_follow_up(cls, text: str) -> Dict[str, Any]:
        res = {"follow_up_date": None, "follow_up_text": None}
        m_fu = re.search(r"(?i)(?:follow\s*up|review|sos\s*review|next\s*visit)[:\s]+([^\n\.;]+)", text)
        if m_fu:
            raw_fu = m_fu.group(1).strip()
            res["follow_up_text"] = raw_fu

            # Check explicit dates like "2026-11-20", "20/11/2026", "Nov 20"
            m_date = re.search(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b", raw_fu)
            if m_date:
                res["follow_up_date"] = m_date.group(1)
            else:
                # Check relative duration like "after 5 days", "in 1 week"
                m_rel = re.search(r"(?i)(?:after|in)?\s*(\d+)\s*(days?|weeks?|months?)", raw_fu)
                if m_rel:
                    num = int(m_rel.group(1))
                    unit = m_rel.group(2).lower()
                    delta = dt.timedelta(days=num if "day" in unit else (num * 7 if "week" in unit else num * 30))
                    target_d = dt.date.today() + delta
                    res["follow_up_date"] = target_d.isoformat()
        return res
