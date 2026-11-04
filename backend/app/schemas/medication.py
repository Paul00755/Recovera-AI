import datetime as dt
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, model_validator

class MedicationScheduleIn(BaseModel):
    label: str = Field(..., max_length=80)
    time_hhmm: str = Field(..., pattern=r"^\d{2}:\d{2}$")
    dose: Optional[str] = None
    food_relation: Optional[str] = "after_food"

class MedicationScheduleOut(BaseModel):
    id: str
    medication_id: Optional[str] = None
    label: str
    time_hhmm: str
    dose: Optional[str] = None
    food_relation: Optional[str] = None

    class Config:
        from_attributes = True

class MedicationCreateIn(BaseModel):
    name: str = Field(..., max_length=200)
    generic_name: Optional[str] = Field(None, max_length=200)
    strength: Optional[str] = Field(None, max_length=100)
    dose: Optional[str] = Field(None, max_length=100)
    unit: Optional[str] = Field(None, max_length=50)
    route: Optional[str] = Field("oral", max_length=50)
    start_date: dt.date
    end_date: dt.date
    instructions: Optional[str] = ""
    source_document_id: Optional[str] = None
    schedules: List[MedicationScheduleIn] = Field(default_factory=list)
    clinical_override_rationale: Optional[str] = None

class InteractionCheckIn(BaseModel):
    patient_id: str
    medication_name: str
    dose: Optional[str] = None

class InteractionItem(BaseModel):
    type: str  # drug_drug, drug_condition, drug_allergy, duplicate_therapy
    severity: str  # contraindicated, severe, moderate
    agent_1: str
    agent_2: str
    risk: str
    recommendation: str

class InteractionCheckOut(BaseModel):
    has_interactions: bool
    max_severity: str
    interactions: List[InteractionItem]
    override_required: bool

class MedicationUpdateIn(BaseModel):
    name: Optional[str] = None
    generic_name: Optional[str] = None
    strength: Optional[str] = None
    dose: Optional[str] = None
    unit: Optional[str] = None
    route: Optional[str] = None
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    instructions: Optional[str] = None
    schedules: Optional[List[MedicationScheduleIn]] = None

class MedicationOut(BaseModel):
    id: str
    patient_id: str
    name: str
    generic_name: Optional[str] = None
    strength: Optional[str] = None
    dose: Optional[str] = None
    unit: Optional[str] = None
    route: Optional[str] = None
    start_date: dt.date
    end_date: dt.date
    instructions: str
    status: str
    version: int
    source_document_id: Optional[str] = None
    created_by: str
    created_at: dt.datetime
    schedules: List[MedicationScheduleOut] = Field(default_factory=list)

    class Config:
        from_attributes = True

class MedicationOccurrenceOut(BaseModel):
    id: str
    medication_id: str
    schedule_id: str
    patient_id: str
    medicine_name: str
    dose: Optional[str] = None
    label: Optional[str] = None
    food_relation: Optional[str] = None
    scheduled_at: dt.datetime
    state: str
    snooze_count: int = 0
    snoozed_until: Optional[dt.datetime] = None
    finalized_at: Optional[dt.datetime] = None
    start_date: Optional[dt.date] = None
    end_date: Optional[dt.date] = None
    adherence: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

class AdherenceRecordIn(BaseModel):
    state: Optional[str] = "taken"
    status: Optional[str] = None
    reason: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def resolve_state(cls, values):
        if isinstance(values, dict):
            if not values.get("state") and values.get("status"):
                values["state"] = values["status"]
            val = values.get("state")
            if val and val not in ("taken", "not_taken"):
                if "taken" in str(val).lower() and "not" not in str(val).lower():
                    values["state"] = "taken"
                else:
                    values["state"] = "not_taken"
            if not values.get("state"):
                values["state"] = "taken"
        return values

class AdherenceRecordOut(BaseModel):
    occurrence_id: str
    state: str
    reason: Optional[str] = None
    recorded_at: dt.datetime
