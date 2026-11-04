import datetime as dt
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field

class RecoveryPlanIn(BaseModel):
    instructions: Optional[Union[Dict[str, Any], List[Any], str]] = None
    warning_signs: List[Any] = Field(default_factory=list)
    status: Optional[str] = "draft"

class RecoveryPlanUpdate(BaseModel):
    instructions: Optional[Union[Dict[str, Any], List[Any], str]] = None
    warning_signs: Optional[List[Any]] = None
    status: Optional[str] = None

class RecoveryPlanOut(BaseModel):
    id: str
    patient_id: str
    version: int
    status: str
    source_versions: Optional[Dict[str, Any]] = Field(default_factory=dict)
    instructions: Optional[Union[Dict[str, Any], List[Any], str]] = None
    warning_signs: List[Any] = Field(default_factory=list)
    approved_by: Optional[str] = None
    approved_at: Optional[dt.datetime] = None
    created_at: dt.datetime
    updated_at: dt.datetime

    class Config:
        from_attributes = True
