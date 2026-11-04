import datetime as dt
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class MedicalHistoryIn(BaseModel):
    has_history: bool = False
    conditions: List[Any] = Field(default_factory=list)
    past_medications: List[Any] = Field(default_factory=list)
    past_treatments: List[Any] = Field(default_factory=list)
    past_hospitalizations: List[Any] = Field(default_factory=list)
    notes: Optional[str] = ""

class PatientCreateIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    dob: Optional[dt.date] = None
    age: Optional[int] = None
    gender: Optional[str] = "unspecified"
    phone: Optional[str] = None
    email: Optional[str] = None
    preferred_language: str = "en"
    sms_language: Optional[str] = "en"
    sms_opt_in: bool = False
    has_medical_history: bool = False
    assigned_doctor_ids: List[str] = Field(default_factory=list)
    medical_history: Optional[MedicalHistoryIn] = None
    password: Optional[str] = None
    must_change_password: bool = True

    @field_validator("dob", mode="before")
    @classmethod
    def clean_empty_dob(cls, v):
        if v == "" or v is None:
            return None
        return v

class PatientUpdateIn(BaseModel):
    name: Optional[str] = None
    dob: Optional[dt.date] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    preferred_language: Optional[str] = None
    sms_language: Optional[str] = None
    sms_opt_in: Optional[bool] = None
    active: Optional[bool] = None
    assigned_doctor_ids: Optional[List[str]] = None
    medical_history: Optional[MedicalHistoryIn] = None

    @field_validator("dob", mode="before")
    @classmethod
    def clean_empty_dob(cls, v):
        if v == "" or v is None:
            return None
        return v

class PatientDeletionTicketCreateIn(BaseModel):
    reason: str = Field(..., min_length=3, max_length=1000)

class PatientDeletionTicketReviewIn(BaseModel):
    admin_notes: Optional[str] = ""

class AssignDoctorRequest(BaseModel):
    doctor_id: str
    department: Optional[str] = None
    notes: Optional[str] = None

class PatientPreferencesIn(BaseModel):
    preferred_language: Optional[str] = None
    sms_language: Optional[str] = None
    sms_opt_in: Optional[bool] = None

class CareStatusUpdateIn(BaseModel):
    care_status: str
    reason: Optional[str] = ""

class PatientConditionUpdateIn(BaseModel):
    condition: Optional[str] = None
    stage: Optional[str] = "recovering"  # acute, recovering, stable, resolved
    notes: Optional[str] = ""

class ReactivationTicketCreateIn(BaseModel):
    reason: str = Field(..., min_length=3, max_length=1000)

class ReactivationTicketReviewIn(BaseModel):
    clinical_notes: Optional[str] = ""

class PatientOut(BaseModel):
    id: str
    patient_code: str
    name: str
    dob: Optional[dt.date] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    preferred_language: str
    sms_opt_in: bool
    has_medical_history: bool
    active: bool
    care_status: Optional[str] = "active"
    active_condition: Optional[str] = None
    condition_stage: Optional[str] = "active"
    condition_notes: Optional[str] = None
    condition_updated_at: Optional[dt.datetime] = None
    created_at: dt.datetime
