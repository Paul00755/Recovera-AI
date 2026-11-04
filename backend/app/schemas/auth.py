from typing import Optional
from pydantic import BaseModel, Field, model_validator

class TokenResponse(BaseModel):
    token: str
    user: dict

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str

class StaffRegisterRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str
    name: str = Field(..., min_length=2, max_length=200)
    phone: str = Field(..., min_length=7, max_length=40)
    role: str = Field(..., pattern="^(doctor|nurse)$")
    staff_code: str
    specialization: Optional[str] = None  # required if role == 'doctor'

    @model_validator(mode="after")
    def validate_staff_fields(self) -> "StaffRegisterRequest":
        if not (self.phone and self.phone.strip()):
            raise ValueError("phone number is mandatory when registering hospital staff.")
        if self.role == "doctor" and not (self.specialization and self.specialization.strip()):
            raise ValueError("specialization is required when registering a doctor.")
        return self


class PatientAccountCreateRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str
    patient_id: str

class ForgotPasswordRequest(BaseModel):
    email: str = Field(..., max_length=320)
    reason: str = Field("Forgot credential", max_length=500)

class ResolveTicketRequest(BaseModel):
    action: str = Field(..., pattern="^(resolved|rejected)$")
    new_password: Optional[str] = None
    notes: Optional[str] = None
    must_change_password: bool = True
    notify_sms: bool = True

class ChangePasswordRequest(BaseModel):
    old_password: Optional[str] = None
    new_password: str
    confirm_password: Optional[str] = None

class ProfileUpdateRequestIn(BaseModel):
    field_name: str = Field(..., min_length=2, max_length=100)
    requested_value: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=3, max_length=1000)

class ProfileUpdateResolveIn(BaseModel):
    action: str = Field(..., pattern="^(approved|rejected)$")
    admin_notes: Optional[str] = None

class AdminUserUpdateIn(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    email: Optional[str] = Field(None, max_length=320)
    phone: Optional[str] = Field(None, max_length=40)
    role: Optional[str] = Field(None, pattern="^(admin|doctor|nurse|patient)$")
    specialization: Optional[str] = None
    availability_status: Optional[str] = Field(None, pattern="^(online|away|busy)$")
    active: Optional[bool] = None
    password: Optional[str] = Field(None, min_length=6)

