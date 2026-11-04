import re
import os
from typing import Dict, Any, List
from pathlib import Path

# Optional PaddleOCR import with fallback
_paddle_ocr_engine = None

def get_ocr_engine():
    global _paddle_ocr_engine
    if _paddle_ocr_engine is None:
        try:
            from paddleocr import PaddleOCR
            # Initialize with English, use_angle_cls=False for speed
            _paddle_ocr_engine = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        except Exception as e:
            # Fallback if paddleocr runtime binaries fail on host
            _paddle_ocr_engine = False
    return _paddle_ocr_engine

def clean_ocr_text(raw_text: str) -> str:
    """
    Cleans raw OCR extraction noise:
    - Removes (page 1), repeated page headers/footers
    - Removes redundant metadata artifacts
    - Normalizes multiple spaces and blank lines while preserving clinical text
    """
    if not raw_text:
        return ""

    text = raw_text

    # Remove page number artifacts like "(page 1)", "Page 1 of 3", "[Page 2]"
    text = re.sub(r"(?i)\(page\s*\d+\)", "", text)
    text = re.sub(r"(?i)\[page\s*\d+\]", "", text)
    text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)
    text = re.sub(r"(?i)page\s+\d+", "", text)

    # Remove common printer/scanner metadata noise
    text = re.sub(r"(?i)scanned\s+by\s+[a-zA-Z0-9\-_]+", "", text)
    text = re.sub(r"(?i)document\s+reference:\s*[a-zA-Z0-9\-]+", "", text)

    # Clean multiple blank lines and whitespace
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.split("\n")]
    cleaned_lines = [line for line in lines if line]

    return "\n".join(cleaned_lines)

def parse_prescription_fields(cleaned_text: str) -> List[Dict[str, Any]]:
    """
    Extracts structured medication candidates from cleaned prescription text:
    - medicine name, strength, dose, route, frequency, food relation
    """
    medications = []
    lines = cleaned_text.split("\n")

    # Common drug forms and routes
    drug_patterns = [
        r"(?i)\b(tab(?:let)?|cap(?:sule)?|syr(?:up)?|inj(?:ection)?|ointment|drops)\b\s+([a-zA-Z0-9\-\s]+)",
        r"(?i)\b([a-zA-Z]{4,20})\s+(\d+\s*(?:mg|ml|mcg|g))\b"
    ]

    for line in lines:
        # Check for medication names with dosages
        for pat in drug_patterns:
            match = re.search(pat, line)
            if match:
                groups = match.groups()
                med_name = groups[1].strip() if len(groups) > 1 else groups[0].strip()
                # Clean prefix words
                med_name = re.sub(r"(?i)^(tab|tablet|cap|capsule|syr|syrup|inj|take)\s+", "", med_name).strip()

                # Extract dosage if present
                dose_match = re.search(r"(\d+\s*(?:mg|ml|mcg|g))", line, re.IGNORECASE)
                dose_val = dose_match.group(1) if dose_match else "1 tablet"

                # Food relation
                food_rel = "after_food" if re.search(r"(?i)after\s+food|post\s+meal", line) else (
                    "before_food" if re.search(r"(?i)before\s+food|pre\s+meal", line) else "with_food"
                )

                # Frequency / timing
                frequency = "twice_daily" if re.search(r"(?i)bd|bid|twice", line) else (
                    "thrice_daily" if re.search(r"(?i)tds|tid|thrice", line) else (
                        "once_daily" if re.search(r"(?i)od|once", line) else "twice_daily"
                    )
                )

                if med_name and len(med_name) >= 3 and not any(m["name"].lower() == med_name.lower() for m in medications):
                    medications.append({
                        "name": med_name.title(),
                        "dose": dose_val,
                        "route": "oral",
                        "frequency": frequency,
                        "food_relation": food_rel,
                        "instructions": line.strip()
                    })

    return medications

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from PDF or image using PyMuPDF / PaddleOCR.
    """
    ext = Path(file_path).suffix.lower()

    # 1. If PDF, first extract embedded text using PyMuPDF (fitz)
    if ext == ".pdf":
        try:
            import fitz
            doc = fitz.open(file_path)
            extracted = []
            for page in doc:
                text = page.get_text()
                if text.strip():
                    extracted.append(text)
            doc.close()
            if extracted:
                return "\n".join(extracted)
        except Exception:
            pass

    # 2. Use PaddleOCR for image or scanned PDF
    engine = get_ocr_engine()
    if engine:
        try:
            result = engine.ocr(file_path, cls=False)
            lines = []
            if result and result[0]:
                for line in result[0]:
                    lines.append(line[1][0])
            return "\n".join(lines)
        except Exception:
            pass

    # 3. Fallback: If text file or mock
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""
