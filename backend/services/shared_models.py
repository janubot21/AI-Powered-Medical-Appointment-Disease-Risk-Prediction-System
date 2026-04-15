from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

class RiskPredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years.")
    gender: str = Field(..., description="Patient gender.")
    symptoms: list[str] | str = Field(..., description="Symptoms as a list or comma-separated string.")
    glucose: float = Field(..., ge=0, description="Glucose reading in mg/dL.")
    glucose_type: Literal["fasting", "random"] = Field(
        ...,
        description="Whether the glucose reading is fasting or random.",
    )
    systolic_bp: float = Field(..., ge=0, description="Systolic blood pressure in mmHg.")
    diastolic_bp: float = Field(..., ge=0, description="Diastolic blood pressure in mmHg.")
    bmi: float = Field(..., ge=0, description="Body Mass Index.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 47,
                    "gender": "Female",
                    "symptoms": ["fatigue", "shortness of breath", "cough", "dizziness"],
                    "glucose": 145.0,
                    "glucose_type": "fasting",
                    "systolic_bp": 148.0,
                    "diastolic_bp": 96.0,
                    "bmi": 31.2,
                },
                {
                    "age": 58,
                    "gender": "Male",
                    "symptoms": ["chest pain", "shortness of breath", "dizziness"],
                    "glucose": 52.0,
                    "glucose_type": "random",
                    "systolic_bp": 190.0,
                    "diastolic_bp": 124.0,
                    "bmi": 33.4,
                },
            ]
        }
    }


class NurseRuleAssessment(BaseModel):
    valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    triage_level: str = "Low"
    priority: str = "Normal booking"
    show_on_top: bool = False
    emergency_alerts: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    appointment_recommendation: str = "Normal booking"
    suggestions: list[str] = Field(default_factory=list)
    nurse_comment: str = ""
    derived: dict[str, Any] = Field(default_factory=dict)


class RiskPredictionResponse(BaseModel):
    risk_group: str
    model_risk_group: str
    clinical_risk_group: str
    risk_probabilities: dict[str, float]
    booking_priority: str
    clinical_assessment: dict[str, Any]
    engineered_features: dict[str, Any]
    nurse_rules: NurseRuleAssessment | None = None


class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100, description="User display name.")
    email: str = Field(..., min_length=5, max_length=254, description="User email address.")
    phone: str = Field(
        ...,
        min_length=10,
        max_length=10,
        pattern=r"^\d{10}$",
        description="User phone number (10 digits).",
    )
    password: str = Field(..., min_length=8, max_length=128, description="Account password.")
    role: Literal["patient", "nurse", "doctor"] = Field(..., description="Account role.")


class LoginRequest(BaseModel):
    email: str | None = Field(None, min_length=5, max_length=254, description="User email address.")
    phone: str | None = Field(
        None,
        min_length=10,
        max_length=10,
        pattern=r"^\d{10}$",
        description="User phone number (10 digits).",
    )
    password: str = Field(..., min_length=6, max_length=128, description="Account password.")
    role: Literal["patient", "nurse", "doctor"] = Field(..., description="Account role.")


class PasswordResetRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)


class PasswordResetConfirm(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)
    token: str = Field(..., min_length=16, max_length=256)
    new_password: str = Field(..., min_length=8, max_length=128)


class HealthDetailsRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254, description="User email address.")
    age: int = Field(..., ge=0, le=120, description="Patient age in years.")
    gender: str = Field(..., min_length=1, max_length=50, description="Patient gender.")
    symptoms: str = Field(..., min_length=1, max_length=1000, description="Symptoms as free text.")
    symptom_count: int = Field(..., ge=0, le=100, description="Total symptom count.")
    glucose: float | None = Field(None, ge=0, description="Glucose level in mg/dL.")
    blood_pressure: str | None = Field(None, max_length=30, description="Blood pressure value, e.g. 120/80.")
    height_cm: float = Field(..., gt=0, le=300, description="Height in centimeters.")
    weight_kg: float = Field(..., gt=0, le=500, description="Weight in kilograms.")
    bmi: float = Field(..., gt=0, le=100, description="Body Mass Index.")
    smoking_habit: str = Field(..., min_length=1, max_length=50, description="Smoking habit.")
    alcohol_habit: str = Field(..., min_length=1, max_length=50, description="Alcohol habit.")
    average_sleep_hours: float = Field(..., ge=0, le=24, description="Average sleep hours.")
    medical_history: str | None = Field(None, max_length=2000, description="Past illnesses, treatment, allergies.")
    family_history_major_chronic_disease: str = Field(..., min_length=1, max_length=100, description="Family history of chronic disease.")


class PatientRecordUpdateRequest(BaseModel):
    patient_email: str = Field(..., min_length=5, max_length=254, description="Patient email address to update.")
    updated_by_email: str = Field(..., min_length=5, max_length=254, description="Editor email (nurse/doctor).")
    updated_by_role: Literal["nurse", "doctor"] = Field("nurse", description="Editor role.")

    age: int | None = Field(None, ge=0, le=120)
    gender: str | None = Field(None, max_length=50)
    symptoms: list[str] | str | None = None
    height_cm: float | None = Field(None, gt=0, le=300)
    weight_kg: float | None = Field(None, gt=0, le=500)
    bmi: float | None = Field(None, gt=0, le=100)

    glucose: float | None = Field(None, ge=0)
    glucose_type: Literal["fasting", "random"] | None = None
    systolic_bp: float | None = Field(None, ge=0)
    diastolic_bp: float | None = Field(None, ge=0)

class AppointmentCreateRequest(BaseModel):
    patient_email: str = Field(..., min_length=5, max_length=254, description="Patient email address.")
    scheduled_for: str = Field(..., min_length=10, max_length=64, description="ISO datetime for the appointment.")
    appointment_type: str = Field(..., min_length=2, max_length=100, description="Appointment type.")
    contact_info: str = Field(..., min_length=3, max_length=200, description="Patient contact info (phone or email).")
    doctor_email: str | None = Field(None, max_length=254, description="Selected doctor email.")
    reason: str | None = Field(None, min_length=2, max_length=200, description="Optional free-text reason.")
    notes: str | None = Field(None, max_length=1000, description="Optional notes.")


class AppointmentCancelRequest(BaseModel):
    patient_email: str = Field(..., min_length=5, max_length=254, description="Patient email address.")
    appointment_id: int = Field(..., ge=1, description="Appointment id to cancel.")
    cancel_reason: str | None = Field(None, max_length=200, description="Optional cancel reason.")


class AppointmentResponse(BaseModel):
    id: int
    patient_email: str
    patient_name: str
    scheduled_for: str
    status: str
    appointment_type: str | None
    doctor_email: str | None
    contact_info: str | None
    priority: str | None
    predicted_risk_level: str | None = None
    predicted_risk_source: str | None = None
    booking_priority: str | None = None
    health_details_submitted_at: str | None = None
    reason: str
    notes: str | None
    created_at: str


class DoctorLeaveRequest(BaseModel):
    doctor_email: str = Field(..., min_length=5, max_length=254, description="Doctor email address.")
    start_at: str = Field(..., min_length=10, max_length=64, description="ISO datetime start of leave.")
    end_at: str = Field(..., min_length=10, max_length=64, description="ISO datetime end of leave.")
    reason: str = Field("", max_length=200, description="Optional leave reason.")

