"""
Recovera AI - Adverse Drug Event (ADE) & Cross-Reactivity Interceptor Service (Phase 13)

Provides comprehensive drug-drug interactions (DDIs), drug-condition contraindications,
and drug-allergy cross-reactivity checks with clinical severity classifications
('contraindicated', 'severe', 'moderate').
"""
import re
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Clinical Knowledge Base
# ---------------------------------------------------------------------------

# Normalized drug classes / aliases
DRUG_ALIASES = {
    # NSAIDs
    "ibuprofen": "nsaid",
    "advil": "nsaid",
    "motrin": "nsaid",
    "naproxen": "nsaid",
    "aleve": "nsaid",
    "diclofenac": "nsaid",
    "voltaren": "nsaid",
    "ketorolac": "nsaid",
    "toradol": "nsaid",
    "indomethacin": "nsaid",
    "meloxicam": "nsaid",
    "aspirin": "aspirin",
    "ecosprin": "aspirin",

    # Anticoagulants / Antiplatelets
    "warfarin": "warfarin",
    "coumadin": "warfarin",
    "clopidogrel": "clopidogrel",
    "plavix": "clopidogrel",
    "apixaban": "doac",
    "eliquis": "doac",
    "rivaroxaban": "doac",
    "xarelto": "doac",
    "heparin": "heparin",

    # ACE inhibitors & ARBs
    "lisinopril": "ace_inhibitor",
    "enalapril": "ace_inhibitor",
    "ramipril": "ace_inhibitor",
    "losartan": "arb",
    "valsartan": "arb",
    "telmisartan": "arb",

    # Potassium sparing / Diuretics
    "spironolactone": "potassium_sparing",
    "aldactone": "potassium_sparing",
    "eplerenone": "potassium_sparing",
    "furosemide": "loop_diuretic",
    "lasix": "loop_diuretic",

    # Beta-blockers
    "propranolol": "nonselective_beta_blocker",
    "carvedilol": "nonselective_beta_blocker",
    "labetalol": "nonselective_beta_blocker",
    "atenolol": "beta_blocker",
    "metoprolol": "beta_blocker",
    "bisoprolol": "beta_blocker",

    # Psychiatric / Neurological
    "sertraline": "ssri",
    "zoloft": "ssri",
    "fluoxetine": "ssri",
    "prozac": "ssri",
    "escitalopram": "ssri",
    "lexapro": "ssri",
    "citalopram": "ssri",
    "paroxetine": "ssri",
    "tramadol": "tramadol",
    "ultram": "tramadol",
    "phenelzine": "maoi",
    "tranylcypromine": "maoi",
    "selegiline": "maoi",

    # Statins & Macrolides
    "simvastatin": "statin_cyp3a4",
    "atorvastatin": "statin_cyp3a4",
    "clarithromycin": "macrolide_cyp3a4_inhibitor",
    "erythromycin": "macrolide_cyp3a4_inhibitor",
    "azithromycin": "macrolide",

    # Diabetes
    "metformin": "metformin",
    "glucophage": "metformin",

    # Cardiac
    "digoxin": "digoxin",
    "lanoxin": "digoxin",
    "amiodarone": "amiodarone",
    "cordarone": "amiodarone",

    # Immunosuppressants
    "methotrexate": "methotrexate",

    # Fluoroquinolones & Steroids
    "ciprofloxacin": "fluoroquinolone",
    "cipro": "fluoroquinolone",
    "levofloxacin": "fluoroquinolone",
    "levaquin": "fluoroquinolone",
    "prednisone": "corticosteroid",
    "dexamethasone": "corticosteroid",
    "hydrocortisone": "corticosteroid",

    # Antibiotics (Beta-lactams, Sulfa)
    "penicillin": "penicillin",
    "amoxicillin": "penicillin",
    "augmentin": "penicillin",
    "ampicillin": "penicillin",
    "cephalexin": "cephalosporin",
    "keflex": "cephalosporin",
    "ceftriaxone": "cephalosporin",
    "cefuroxime": "cephalosporin",
    "bactrim": "sulfa_antibiotic",
    "septra": "sulfa_antibiotic",
    "sulfamethoxazole": "sulfa_antibiotic",

    # PPI
    "omeprazole": "omeprazole",
    "prilosec": "omeprazole",
    "esomeprazole": "omeprazole",
    "nexium": "omeprazole",
}

# ---------------------------------------------------------------------------
# Drug-Drug Interaction Rules
# ---------------------------------------------------------------------------
DDI_RULES = [
    {
        "agents": ({"warfarin", "doac"}, {"nsaid", "aspirin"}),
        "severity": "contraindicated",
        "risk": "Major gastrointestinal bleeding and life-threatening systemic hemorrhage.",
        "recommendation": "Avoid concurrent use. Use alternative analgesia such as acetaminophen or prescribe gastroprotective agent (PPI) with frequent INR/coagulation monitoring."
    },
    {
        "agents": ({"ace_inhibitor", "arb"}, {"potassium_sparing"}),
        "severity": "severe",
        "risk": "Severe hyperkalemia leading to cardiac arrhythmias and cardiac arrest.",
        "recommendation": "Monitor serum potassium and creatinine within 1-2 weeks of initiation. Advise patient against potassium-rich supplements."
    },
    {
        "agents": ({"ssri"}, {"tramadol", "maoi"}),
        "severity": "contraindicated",
        "risk": "Serotonin syndrome characterized by tremor, hyperthermia, clonus, autonomic instability, and potential fatality.",
        "recommendation": "Contraindicated. Discontinue one agent or select a non-serotonergic analgesic or antidepressant."
    },
    {
        "agents": ({"statin_cyp3a4"}, {"macrolide_cyp3a4_inhibitor"}),
        "severity": "severe",
        "risk": "Potent CYP3A4 inhibition increases statin serum concentrations dramatically, causing severe rhabdomyolysis and acute renal failure.",
        "recommendation": "Temporarily suspend simvastatin/atorvastatin during macrolide therapy or substitute with azithromycin or non-macrolide antibiotic."
    },
    {
        "agents": ({"digoxin"}, {"amiodarone"}),
        "severity": "severe",
        "risk": "Amiodarone reduces renal and non-renal clearance of digoxin, doubling serum digoxin levels and precipitating digitalis toxicity (arrhythmias, heart block).",
        "recommendation": "Reduce digoxin dosage by 50% upon initiating amiodarone and monitor serum digoxin levels closely."
    },
    {
        "agents": ({"methotrexate"}, {"nsaid", "aspirin"}),
        "severity": "severe",
        "risk": "NSAIDs diminish methotrexate renal clearance, causing profound myelosuppression and severe bone marrow failure.",
        "recommendation": "Avoid high-dose methotrexate co-administration with NSAIDs. Monitor CBC and renal function closely."
    },
    {
        "agents": ({"fluoroquinolone"}, {"corticosteroid"}),
        "severity": "severe",
        "risk": "Synergistic elevation of tendon rupture risk, particularly Achilles tendon, especially in elderly patients.",
        "recommendation": "Avoid combination when possible. Instruct patient to report tendon pain or inflammation immediately and discontinue fluoroquinolone."
    },
    {
        "agents": ({"clopidogrel"}, {"omeprazole"}),
        "severity": "moderate",
        "risk": "Omeprazole inhibits CYP2C19 bioactivation of clopidogrel, attenuating its antiplatelet effect and increasing thrombotic event risk.",
        "recommendation": "Substitute omeprazole with pantoprazole, famotidine, or another H2 blocker with negligible CYP2C19 inhibition."
    },
]

# ---------------------------------------------------------------------------
# Drug-Condition Contraindications
# ---------------------------------------------------------------------------
DRUG_CONDITION_RULES = [
    {
        "drug_types": {"nsaid"},
        "condition_keywords": ["ckd", "chronic kidney disease", "renal failure", "renal impairment", "kidney failure", "nephropathy"],
        "severity": "contraindicated",
        "risk": "NSAIDs inhibit renal prostaglandins, precipitating acute renal decompensation, fluid retention, and progressive CKD stage advancement.",
        "recommendation": "Contraindicated in moderate-to-severe renal impairment. Use acetaminophen or topical non-systemic analgesics."
    },
    {
        "drug_types": {"nsaid", "aspirin"},
        "condition_keywords": ["peptic ulcer", "gastric ulcer", "gi bleed", "gastrointestinal bleeding", "duodenal ulcer", "ulcerative colitis"],
        "severity": "contraindicated",
        "risk": "Direct mucosal injury and cyclooxygenase-1 inhibition precipitating recurrent gastrointestinal ulceration and hemorrhage.",
        "recommendation": "Strictly contraindicated without robust gastroprotection. Consider COX-2 selective agent plus PPI or acetaminophen."
    },
    {
        "drug_types": {"nonselective_beta_blocker", "beta_blocker"},
        "condition_keywords": ["asthma", "copd", "bronchospasm", "reactive airway"],
        "severity": "severe",
        "risk": "Beta-2 receptor blockade causes severe bronchoconstriction, potentially triggering fatal status asthmaticus.",
        "recommendation": "Non-selective beta-blockers are contraindicated in asthma. If beta-blockade is essential, use cardioselective beta-1 agents with caution."
    },
    {
        "drug_types": {"metformin"},
        "condition_keywords": ["severe renal", "ckd stage 4", "ckd stage 5", "esrd", "dialysis", "lactic acidosis", "severe liver failure", "cirrhosis"],
        "severity": "contraindicated",
        "risk": "Diminished metformin clearance precipitates accumulation and fatal lactic acidosis.",
        "recommendation": "Discontinue metformin if eGFR falls below 30 mL/min/1.73m2. Shift to insulin or renal-sparing antidiabetic therapy."
    },
    {
        "drug_types": {"ace_inhibitor", "arb"},
        "condition_keywords": ["pregnancy", "pregnant", "gestation"],
        "severity": "contraindicated",
        "risk": "Black Box Warning: Fetopathy, neonatal renal failure, oligohydramnios, skull hypoplasia, and fetal death.",
        "recommendation": "Strictly contraindicated in pregnancy. Discontinue immediately and transition to labetalol, nifedipine, or methyldopa."
    },
]

# ---------------------------------------------------------------------------
# Drug-Allergy Cross-Reactivity Rules
# ---------------------------------------------------------------------------
DRUG_ALLERGY_RULES = [
    {
        "drug_types": {"penicillin"},
        "allergy_keywords": ["penicillin", "amoxicillin", "ampicillin", "augmentin", "beta-lactam"],
        "severity": "contraindicated",
        "risk": "Type I IgE-mediated anaphylaxis, angioedema, urticaria, or severe cutaneous adverse reactions (SCAR).",
        "recommendation": "Strictly contraindicated. Switch to alternative class such as macrolides, fluoroquinolones, or clindamycin."
    },
    {
        "drug_types": {"cephalosporin"},
        "allergy_keywords": ["penicillin", "amoxicillin", "ampicillin"],
        "severity": "severe",
        "risk": "Potential cross-reactivity (up to 5-10%) between beta-lactam rings and similar R1 side chains leading to allergic reactions.",
        "recommendation": "Use 3rd or 4th generation cephalosporins with distinct side chains or perform skin allergy testing. Avoid in patients with history of anaphylaxis."
    },
    {
        "drug_types": {"sulfa_antibiotic"},
        "allergy_keywords": ["sulfa", "sulfonamide", "bactrim", "septra"],
        "severity": "contraindicated",
        "risk": "Severe hypersensitivity syndrome, Stevens-Johnson syndrome (SJS), or toxic epidermal necrolysis (TEN).",
        "recommendation": "Strictly contraindicated. Select non-sulfonamide antimicrobial coverage."
    },
    {
        "drug_types": {"nsaid"},
        "allergy_keywords": ["aspirin", "nsaid", "ibuprofen", "naproxen"],
        "severity": "contraindicated",
        "risk": "Aspirin-exacerbated respiratory disease (AERD), bronchospasm, or angioedema due to cyclooxygenase inhibition and leukotriene shunting.",
        "recommendation": "Strictly contraindicated. Utilize acetaminophen or opioid analgesics with desensitization if necessary."
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^\w\s]", "", text.strip().lower())

def resolve_drug_type(drug_name: str) -> str:
    """Resolves a drug name or brand to a normalized drug class identifier."""
    cleaned = normalize_text(drug_name)
    tokens = cleaned.split()
    for token in tokens:
        if token in DRUG_ALIASES:
            return DRUG_ALIASES[token]
    # Check substring matches
    for brand, generic in DRUG_ALIASES.items():
        if brand in cleaned:
            return generic
    return cleaned

def extract_strings_from_container(container: Any) -> List[str]:
    """Safely extracts text strings from list of strings, dicts, or strings."""
    result = []
    if isinstance(container, str):
        result.append(container)
    elif isinstance(container, list):
        for item in container:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                # Common keys in condition or medication dicts
                for key in ["name", "condition", "title", "label", "text", "allergy"]:
                    if key in item and item[key]:
                        result.append(str(item[key]))
                        break
                else:
                    result.append(" ".join(str(v) for v in item.values()))
    elif isinstance(container, dict):
        for v in container.values():
            result.extend(extract_strings_from_container(v))
    return result

# ---------------------------------------------------------------------------
# Interceptor Logic
# ---------------------------------------------------------------------------

def check_drug_interactions(
    candidate_medication: str,
    candidate_dose: Optional[str] = None,
    active_medications: Optional[List[str]] = None,
    conditions: Optional[List[Any]] = None,
    allergies: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Performs multi-dimensional ADE checks for a candidate drug:
    1. Drug-Drug Interactions (against active medications)
    2. Drug-Condition Contraindications (against active patient conditions)
    3. Drug-Allergy Cross-Reactivity (against documented allergies)
    """
    active_medications = active_medications or []
    conditions = conditions or []
    allergies = allergies or []

    candidate_type = resolve_drug_type(candidate_medication)
    interactions: List[Dict[str, Any]] = []

    # 1. Drug-Drug Interactions
    for active_med in active_medications:
        active_type = resolve_drug_type(active_med)
        if not active_type or active_type == candidate_type:
            # Check duplicate therapy
            if active_type and active_type == candidate_type:
                interactions.append({
                    "type": "duplicate_therapy",
                    "severity": "moderate",
                    "agent_1": candidate_medication,
                    "agent_2": active_med,
                    "risk": f"Duplicate therapeutic class ({candidate_type}). Review for possible therapeutic redundancy.",
                    "recommendation": "Review whether intentional therapy modification or accidental duplicate prescription."
                })
            continue

        for rule in DDI_RULES:
            set_a, set_b = rule["agents"]
            if (candidate_type in set_a and active_type in set_b) or (candidate_type in set_b and active_type in set_a):
                interactions.append({
                    "type": "drug_drug",
                    "severity": rule["severity"],
                    "agent_1": candidate_medication,
                    "agent_2": active_med,
                    "risk": rule["risk"],
                    "recommendation": rule["recommendation"]
                })

    # 2. Drug-Condition Contraindications
    flat_conditions = [normalize_text(c) for c in extract_strings_from_container(conditions)]
    for cond_text in flat_conditions:
        if not cond_text:
            continue
        for rule in DRUG_CONDITION_RULES:
            if candidate_type in rule["drug_types"]:
                for kw in rule["condition_keywords"]:
                    if kw in cond_text:
                        interactions.append({
                            "type": "drug_condition",
                            "severity": rule["severity"],
                            "agent_1": candidate_medication,
                            "agent_2": cond_text.title(),
                            "risk": rule["risk"],
                            "recommendation": rule["recommendation"]
                        })
                        break

    # 3. Drug-Allergy Cross-Reactivity
    flat_allergies = [normalize_text(a) for a in extract_strings_from_container(allergies)]
    for allergy_text in flat_allergies:
        if not allergy_text:
            continue
        for rule in DRUG_ALLERGY_RULES:
            if candidate_type in rule["drug_types"]:
                for kw in rule["allergy_keywords"]:
                    if kw in allergy_text:
                        interactions.append({
                            "type": "drug_allergy",
                            "severity": rule["severity"],
                            "agent_1": candidate_medication,
                            "agent_2": allergy_text.title(),
                            "risk": rule["risk"],
                            "recommendation": rule["recommendation"]
                        })
                        break

    # Determine highest severity
    severities = [item["severity"] for item in interactions]
    max_severity = "none"
    if "contraindicated" in severities:
        max_severity = "contraindicated"
    elif "severe" in severities:
        max_severity = "severe"
    elif "moderate" in severities:
        max_severity = "moderate"

    # Require clinical override if contraindicated or severe
    override_required = max_severity in ["contraindicated", "severe"]

    return {
        "has_interactions": len(interactions) > 0,
        "max_severity": max_severity,
        "interactions": interactions,
        "override_required": override_required
    }
