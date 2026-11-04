import datetime as dt
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class DocumentReviewPatch(BaseModel):
    extracted_text: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None  # draft, under_review, approved, rejected
    assigned_doctor_id: Optional[str] = None

class DocumentOut(BaseModel):
    id: str
    patient_id: str
    doc_type: str
    filename: str
    status: str
    version: int
    extracted_text: str
    structured_data: Dict[str, Any]
    uploaded_by: str
    reviewed_by: Optional[str] = None
    created_at: dt.datetime
    updated_at: dt.datetime
