from __future__ import annotations

import ast
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from app_config import COOKIE_SECURE, OPENAI_API_KEY, OPENAI_CHAT_MODEL, get_flask_secret_key
from doctor_auth import DoctorAuthManager
from nurse_auth import NurseAuthManager
from patient_auth import PatientAuthManager
from paths import (
    DOCTOR_ACCOUNTS_CSV,
    DOCTOR_LEAVE_CSV,
    DOCTOR_PROFILE_CSV,
    NEW_PATIENT_CSV,
    NURSE_ACCOUNTS_CSV,
    PATIENTS_CSV,
    PATIENT_DB_PATH,
    ensure_csv_exists,
)
from patient_db import PatientDatabase
from portal_auth import (
    _clear_doctor_session,
    _clear_nurse_session,
    _clear_patient_session,
    _is_doctor_authenticated,
    _is_patient_authenticated,
    doctor_required,
    nurse_required,
    patient_required,
)
from predict import LABEL_ENCODER_PATH, MODEL_PATH, RiskEngine, to_jsonable
from triage import determine_priority


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
    static_url_path="/static",
)
app.secret_key = get_flask_secret_key()
app.config.update(
    TEMPLATES_AUTO_RELOAD=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=COOKIE_SECURE,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)
app.jinja_env.auto_reload = True

doctor_auth_manager = DoctorAuthManager()
nurse_auth_manager = NurseAuthManager()
patient_auth_manager = PatientAuthManager()
risk_engine = RiskEngine(MODEL_PATH, LABEL_ENCODER_PATH)
patient_db = PatientDatabase(PATIENT_DB_PATH)
PASSWORD_POLICY_PATTERN = re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")
PASSWORD_POLICY_MESSAGE = (
    "Password must contain minimum 8 characters, including uppercase (A-Z), lowercase (a-z), number (0-9), and special character (@,!,#,$,%,&,*)."
)
SYMPTOMS_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9,\s./()+:&'-]+")
CONTACT_INFO_PATTERN = re.compile(r"^\+?[0-9][0-9\s-]{7,14}$")
HEALTH_DATA_SUBMITTED_AT_COLUMN = "Health_Data_Submitted_At"
_openai_chat_client: Any = None
_openai_chat_client_ready = False
CHAT_GENERAL_PATTERNS = (
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "who are you",
    "what can you do",
)
SAFE_MATH_OPERATORS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: a,
}


def _parse_appointment_time(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_booked_at(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _datetime_sort_value(value: Optional[datetime], *, default: float = float("inf")) -> float:
    if value is None:
        return default
    try:
        return value.timestamp()
    except Exception:
        return default


def _appointment_day(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return value.replace(hour=0, minute=0, second=0, microsecond=0)


def _is_upcoming_appointment_date(value: Optional[datetime], *, reference: Optional[datetime] = None) -> bool:
    appointment_day = _appointment_day(value)
    if appointment_day is None:
        return False
    reference_day = _appointment_day(reference or datetime.now())
    return bool(reference_day and appointment_day >= reference_day)


def _appointment_history_status(value: Optional[datetime], *, reference: Optional[datetime] = None) -> str:
    return "Completed" if value and not _is_upcoming_appointment_date(value, reference=reference) else "Missed"


def _normalize_risk_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "high" in text:
        return "High Risk"
    if "medium" in text:
        return "Medium Risk"
    return "Low Risk"


def _risk_rank(value: Any) -> int:
    label = _normalize_risk_label(value)
    if label == "High Risk":
        return 0
    if label == "Medium Risk":
        return 1
    return 2


def _priority_rank(value: Any) -> int:
    text = str(value or "").strip().lower()
    if "immediate" in text:
        return 0
    if "same" in text:
        return 1
    return 2


def _doctor_appointment_sort_key(item: Dict[str, Any]) -> tuple[int, datetime, float]:
    priority_rank = _priority_rank(item.get("appointment_priority"))
    risk_raw = (item.get("risk_assessment") or {}).get("predicted_class")
    rank = _risk_rank(risk_raw)
    at = _parse_appointment_time(item.get("appointment_time")) or datetime.max
    booked_at = str(item.get("booked_at", "")).strip()
    try:
        booked_ts = datetime.fromisoformat(booked_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        booked_ts = 0.0
    # Explicit triage priority first; fallback to risk class; then earliest slot; then newest booking.
    return (priority_rank, rank, at, -booked_ts)


def _format_date_label(dt: Optional[datetime]) -> str:
    if dt is None:
        return "--"
    return dt.strftime("%A, %B %d, %Y")


def _format_time_label(dt: Optional[datetime]) -> str:
    if dt is None:
        return "--"
    start = dt.strftime("%I:%M %p").lstrip("0")
    end = (dt.replace(minute=0, second=0, microsecond=0) if dt.minute == 0 else dt).replace(second=0, microsecond=0)
    end = end.replace(hour=end.hour + 1) if end.hour < 23 else end
    end_text = end.strftime("%I:%M %p").lstrip("0")
    return f"{start} - {end_text}"


def _format_time_value(dt: Optional[datetime]) -> str:
    if dt is None:
        return "--"
    return dt.strftime("%I:%M %p").lstrip("0")


def _to_int(value: str, field_name: str, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid integer.") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}.")
    return parsed


def _to_float(value: str, field_name: str, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}.")
    return parsed


def _normalize_contact_info(value: Any) -> str:
    text = str(value or "").strip()
    if not CONTACT_INFO_PATTERN.fullmatch(text):
        raise ValueError("contact_info must be a valid phone number.")
    return text


def _ensure_future_appointment_time(value: datetime) -> datetime:
    if value <= datetime.now(value.tzinfo):
        raise ValueError("appointment_time must be in the future.")
    return value


def _normalize_average_sleep_hours(value: Any, *, required: bool = False) -> Optional[int]:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError("Average Sleep Hours is required.")
        return None
    if any(ch in text.lower() for ch in ("e", "+", "-", ".")):
        raise ValueError("Average Sleep Hours must be a whole number between 3 and 12.")
    parsed = _to_int(text, "Average Sleep Hours", 3, 12)
    return parsed


def _sleep_category_details(value: Any) -> tuple[str, str]:
    hours = _to_float_or_none(value)
    if hours is None:
        return ("", "")
    if hours < 5:
        return ("Poor Sleep", "High health risk")
    if hours <= 6:
        return ("Low Sleep", "Not enough rest")
    if hours <= 9:
        return ("Healthy Sleep", "Recommended")
    return ("Excess Sleep", "Possible fatigue")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_submission_timestamp(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return _utc_now_iso()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).isoformat(timespec="seconds").replace("+00:00", "Z")
    except ValueError:
        return _utc_now_iso()


def _rows_match_except_submission_timestamp(left: Dict[str, Any], right: Dict[str, Any], columns: list[str]) -> bool:
    for col in columns:
        if col == HEALTH_DATA_SUBMITTED_AT_COLUMN:
            continue
        left_val = left.get(col)
        right_val = right.get(col)
        left_text = "" if pd.isna(left_val) else str(left_val).strip()
        right_text = "" if pd.isna(right_val) else str(right_val).strip()
        if left_text != right_text:
            return False
    return True


def append_to_new_patient_csv(row: Dict[str, Any], csv_path: Path = NEW_PATIENT_CSV) -> None:
    ensure_csv_exists(csv_path)
    row = dict(row)
    row[HEALTH_DATA_SUBMITTED_AT_COLUMN] = _normalize_submission_timestamp(row.get(HEALTH_DATA_SUBMITTED_AT_COLUMN))
    new_row_df = pd.DataFrame([row])
    try:
        existing_df = pd.read_csv(csv_path)
    except Exception:
        existing_df = pd.DataFrame(columns=list(row.keys()))
    if HEALTH_DATA_SUBMITTED_AT_COLUMN not in existing_df.columns:
        existing_df[HEALTH_DATA_SUBMITTED_AT_COLUMN] = pd.NA
    updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    updated_df.to_csv(csv_path, index=False)
    patient_name = _get_patient_name_by_id(str(row.get("Patient_ID", "")).strip())
    _save_patient_profile_to_db(row, patient_name)


def _gender_to_csv_value(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"1", "male", "m"}:
        return "male"
    if raw in {"0", "female", "f"}:
        return "female"
    if raw in {"-1", "other"}:
        return "other"
    return "male"


def _yes_no_to_csv_value(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y"}:
        return "yes"
    if raw in {"0", "false", "no", "n"}:
        return "no"
    return "no"


def _to_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_systolic_bp(value: Any) -> Optional[float]:
    direct = _to_float_or_none(value)
    if direct is not None:
        return direct

    text = str(value).strip()
    if not text:
        return None
    parts = re.split(r"[/\s]+", text)
    if not parts:
        return None
    return _to_float_or_none(parts[0])


def _extract_diastolic_bp(value: Any) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    parts = re.split(r"[/\s]+", text)
    if len(parts) < 2:
        return None
    return _to_float_or_none(parts[1])


def _safe_db_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _save_patient_profile_to_db(row: Dict[str, Any], patient_name: str = "") -> None:
    patient_id = _safe_db_text(row.get("Patient_ID"))
    if not patient_id:
        return
    bmi = _to_float_or_none(row.get("BMI"))
    payload = {
        "patient_id": patient_id,
        "patient_name": _safe_db_text(patient_name),
        "age": _to_float_or_none(row.get("Age")),
        "gender": _safe_db_text(row.get("Gender")),
        "symptoms": _safe_db_text(row.get("Symptoms")),
        "symptom_count": _to_float_or_none(row.get("Symptom_Count")),
        "glucose": _to_float_or_none(row.get("Glucose")),
        "height_cm": _to_float_or_none(row.get("Height_cm")),
        "weight_kg": _to_float_or_none(row.get("Weight_kg")),
        "weight_category": _safe_db_text(row.get("BMI_Category")) or patient_db.weight_category_from_bmi(bmi),
        "calculated_bmi": bmi,
        "blood_pressure_systolic": _extract_systolic_bp(row.get("BloodPressure")),
        "blood_pressure_diastolic": _extract_diastolic_bp(row.get("BloodPressure")),
        "smoking_habit": _safe_db_text(row.get("Smoking_Habit")),
        "alcohol_habit": _safe_db_text(row.get("Alcohol_Habit")),
        "average_sleep_hours": _to_float_or_none(row.get("Average_Sleep_Hours")),
        "family_history": _safe_db_text(row.get("Family_History")),
        "medical_history": _safe_db_text(row.get("Medical_History")),
        "health_data_submitted_at": _normalize_submission_timestamp(row.get(HEALTH_DATA_SUBMITTED_AT_COLUMN)),
        "nurse_updated_by": _safe_db_text(row.get("Nurse_Updated_By")),
        "nurse_notes": _safe_db_text(row.get("Nurse_Notes")),
        "doctor_reviewed_by": _safe_db_text(row.get("Doctor_Reviewed_By")),
        "reviewed_at": _safe_db_text(row.get("Reviewed_At")),
    }
    patient_db.upsert_profile(payload)


def _build_csv_row_from_features(patient_id: str, patient_features: Dict[str, Any]) -> Dict[str, Any]:
    symptom_count = patient_features.get("Symptom_Count", patient_features.get("Sympton_Count"))
    normalized_pid = _normalize_patient_id(patient_id)
    try:
        patient_id_cell: Any = int(normalized_pid)
    except (TypeError, ValueError):
        patient_id_cell = normalized_pid
    row = {
        "Patient_ID": patient_id_cell,
        "Age": _to_float_or_none(patient_features.get("Age")),
        "Gender": _gender_to_csv_value(patient_features.get("Gender")),
        "Symptoms": str(patient_features.get("Symptoms", "")).strip(),
        "Symptom_Count": _to_float_or_none(symptom_count),
        "Glucose": _to_float_or_none(patient_features.get("Glucose")),
        "BloodPressure": _extract_systolic_bp(patient_features.get("BloodPressure")),
        "BMI": _to_float_or_none(patient_features.get("BMI")),
        "Height_cm": _to_float_or_none(patient_features.get("Height_cm")),
        "Weight_kg": _to_float_or_none(patient_features.get("Weight_kg")),
        "BMI_Category": str(patient_features.get("BMI_Category", "")).strip(),
        "Smoking_Habit": _yes_no_to_csv_value(patient_features.get("Smoking_Habit")),
        "Alcohol_Habit": _yes_no_to_csv_value(patient_features.get("Alcohol_Habit")),
        "Average_Sleep_Hours": _normalize_average_sleep_hours(patient_features.get("Average_Sleep_Hours")),
        "Medical_History": str(patient_features.get("Medical_History", "")).strip(),
        "Family_History": _yes_no_to_csv_value(patient_features.get("Family_History")),
        "Nurse_Updated_By": str(patient_features.get("Nurse_Updated_By", "")).strip(),
        "Nurse_Notes": str(patient_features.get("Nurse_Notes", "")).strip(),
        "Doctor_Reviewed_By": str(patient_features.get("Doctor_Reviewed_By", "")).strip(),
        "Reviewed_At": str(patient_features.get("Reviewed_At", "")).strip(),
        HEALTH_DATA_SUBMITTED_AT_COLUMN: _normalize_submission_timestamp(
            patient_features.get(HEALTH_DATA_SUBMITTED_AT_COLUMN) or patient_features.get("health_data_submitted_at")
        ),
    }
    return row


def upsert_new_patient_csv_from_features(
    patient_id: str, patient_features: Dict[str, Any], csv_path: Path = NEW_PATIENT_CSV
) -> None:
    columns = [
        "Patient_ID",
        "Age",
        "Gender",
        "Symptoms",
        "Symptom_Count",
        "Glucose",
        "BloodPressure",
        "BMI",
        "Height_cm",
        "Weight_kg",
        "BMI_Category",
        "Smoking_Habit",
        "Alcohol_Habit",
        "Average_Sleep_Hours",
        "Medical_History",
        "Family_History",
        "Nurse_Updated_By",
        "Nurse_Notes",
        "Doctor_Reviewed_By",
        "Reviewed_At",
        HEALTH_DATA_SUBMITTED_AT_COLUMN,
    ]
    row = _build_csv_row_from_features(patient_id, patient_features)

    ensure_csv_exists(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
        # Use object dtype to avoid pandas dtype write errors on mixed old/new values.
        df[col] = df[col].astype("object")

    pid = _normalize_patient_id(patient_id)
    matches = df.index[df["Patient_ID"].astype(str).map(_normalize_patient_id) == pid].tolist()

    if matches:
        target_idx = matches[-1]
        # Preserve any previous values when edited payload is partial.
        existing = df.loc[target_idx].to_dict()
        merged = dict(existing)
        for col in columns:
            new_val = row.get(col)
            if new_val is None and col != "Patient_ID":
                continue
            merged[col] = new_val
        if _rows_match_except_submission_timestamp(existing, merged, columns):
            merged[HEALTH_DATA_SUBMITTED_AT_COLUMN] = existing.get(HEALTH_DATA_SUBMITTED_AT_COLUMN)
        else:
            merged[HEALTH_DATA_SUBMITTED_AT_COLUMN] = _utc_now_iso()
        for col in columns:
            df.at[target_idx, col] = merged.get(col)
    else:
        df = pd.concat([df, pd.DataFrame([{col: row.get(col) for col in columns}])], ignore_index=True)

    df.to_csv(csv_path, index=False)
    _save_patient_profile_to_db(row, _get_patient_name_by_id(patient_id))


PATIENTS_DF_COLUMNS = ["patient_id", "name", "unique_code", "health_details_submitted", "created_at"]
LEGACY_PATIENT_PASSWORD_COLUMN = "password"


def _save_patients_df(df: pd.DataFrame) -> None:
    normalized = df.copy()
    for col in PATIENTS_DF_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = ""
    normalized = normalized[PATIENTS_DF_COLUMNS]
    normalized.to_csv(PATIENTS_CSV, index=False)


def _migrate_legacy_patient_passwords(df: pd.DataFrame) -> pd.DataFrame:
    if LEGACY_PATIENT_PASSWORD_COLUMN not in df.columns:
        return df

    migrated_count = 0
    for _, row in df.iterrows():
        patient_id = str(row.get("patient_id", "")).strip()
        legacy_password = str(row.get(LEGACY_PATIENT_PASSWORD_COLUMN, "")).strip()
        if not patient_id or not legacy_password:
            continue
        if patient_auth_manager.account_exists(patient_id):
            continue
        patient_auth_manager.import_plaintext_password(patient_id, legacy_password)
        migrated_count += 1

    if migrated_count:
        logger.warning("Migrated %s legacy patient account(s) from plaintext storage to hashed credentials.", migrated_count)

    df = df.drop(columns=[LEGACY_PATIENT_PASSWORD_COLUMN], errors="ignore")
    return df


def _load_patients_df() -> pd.DataFrame:
    ensure_csv_exists(PATIENTS_CSV)
    try:
        df = pd.read_csv(PATIENTS_CSV)
    except Exception:
        return pd.DataFrame(columns=PATIENTS_DF_COLUMNS)
    df = _migrate_legacy_patient_passwords(df)
    for col in PATIENTS_DF_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    cleaned = df[PATIENTS_DF_COLUMNS]
    _save_patients_df(cleaned)
    return cleaned


def _get_patient_name_by_id(patient_id: str) -> str:
    if not patient_id:
        return ""
    df = _load_patients_df()
    if df.empty:
        return ""
    matches = df[df["patient_id"].astype(str).str.strip() == str(patient_id).strip()]
    if matches.empty:
        return ""
    return str(matches.iloc[-1].get("name", "")).strip()


def _next_patient_id(df: pd.DataFrame) -> str:
    if df.empty:
        return "1"
    ids = pd.to_numeric(df["patient_id"], errors="coerce").dropna()
    if ids.empty:
        return "1"
    return str(int(ids.max()) + 1)


def _normalize_unique_code(value: Any) -> str:
    return str(value).strip().upper()


def _normalize_id_type(value: Any) -> str:
    raw = str(value).strip().lower()
    if raw not in {"aadhaar", "pan"}:
        raise ValueError("ID Type must be Aadhaar or PAN.")
    return raw


def _normalize_id_number(id_type: str, value: Any) -> str:
    raw = str(value).strip().upper()
    if id_type == "aadhaar":
        if not re.fullmatch(r"[0-9]{12}", raw):
            raise ValueError("Aadhaar number must be exactly 12 digits.")
        return raw
    if not re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", raw):
        raise ValueError("PAN must be in format: AAAAA9999A.")
    return raw


def _compose_unique_code(id_type: Any, id_number: Any) -> str:
    norm_type = _normalize_id_type(id_type)
    norm_number = _normalize_id_number(norm_type, id_number)
    return f"{norm_type.upper()}:{norm_number}"


def _create_patient_account(name: str, unique_code: str, password: str) -> str:
    if not PASSWORD_POLICY_PATTERN.fullmatch(str(password).strip()):
        raise ValueError(PASSWORD_POLICY_MESSAGE)
    df = _load_patients_df()
    if df["name"].astype(str).str.strip().str.lower().eq(name.lower()).any():
        raise ValueError("Patient name already exists. Use a different name.")
    patient_id = _next_patient_id(df)
    normalized_code = _normalize_unique_code(unique_code) if str(unique_code).strip() else f"AUTO:PATIENT-{patient_id}"
    existing_codes = df["unique_code"].astype(str).str.strip().str.upper()
    if (existing_codes == normalized_code).any() or (existing_codes == normalized_code.split(":", 1)[-1]).any():
        raise ValueError("Unique Code already exists. Use a different Unique Code.")
    new_row = {
        "patient_id": patient_id,
        "name": name,
        "unique_code": normalized_code,
        "health_details_submitted": 0,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    patient_auth_manager.signup(patient_id, password)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_patients_df(df)
    return patient_id


def _authenticate_patient(name: str, unique_code: str, password: str) -> Optional[Dict[str, Any]]:
    df = _load_patients_df()
    name = str(name or "").strip()
    unique_code = str(unique_code or "").strip()
    has_name = bool(name)
    has_code = bool(unique_code)
    if not has_name and not has_code:
        return None

    identifier_mask = pd.Series([False] * len(df), index=df.index)
    if has_name:
        identifier_mask = identifier_mask | (df["name"].astype(str).str.strip().str.lower() == name.lower())
    if has_code:
        normalized_code = _normalize_unique_code(unique_code)
        identifier_mask = identifier_mask | (
            (df["unique_code"].astype(str).str.strip().str.upper() == normalized_code)
            | (df["unique_code"].astype(str).str.strip().str.upper() == normalized_code.split(":", 1)[-1])
        )

    matches = df[identifier_mask]
    if matches.empty:
        return None
    patient = matches.iloc[-1].to_dict()
    patient_id = str(patient.get("patient_id", "")).strip()
    if not patient_id:
        return None

    try:
        patient_auth_manager.login(patient_id, password)
    except ValueError:
        return None

    return patient


def _mark_health_details_submitted(patient_id: str) -> None:
    df = _load_patients_df()
    if df.empty:
        return
    mask = df["patient_id"].astype(str).str.strip() == str(patient_id).strip()
    if mask.any():
        df.loc[mask, "health_details_submitted"] = 1
        _save_patients_df(df)


def _health_details_submitted(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes"}


def _load_doctor_ids() -> list[str]:
    ensure_csv_exists(DOCTOR_ACCOUNTS_CSV)
    try:
        df = pd.read_csv(DOCTOR_ACCOUNTS_CSV)
    except Exception:
        return []
    if "doctor_id" not in df.columns:
        return []
    doctor_ids = [str(v).strip() for v in df["doctor_id"].tolist() if str(v).strip()]
    return sorted(set(doctor_ids))


def _load_nurse_ids() -> list[str]:
    try:
        ensure_csv_exists(NURSE_ACCOUNTS_CSV)
        df = pd.read_csv(NURSE_ACCOUNTS_CSV)
    except Exception:
        return []
    if "nurse_id" not in df.columns:
        return []
    nurse_ids = [str(v).strip() for v in df["nurse_id"].tolist() if str(v).strip()]
    return sorted(set(nurse_ids))


def _empty_nurse_record(patient_id: str, patient_name: str = "") -> Dict[str, Any]:
    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "appointment_id": "",
        "appointment_time": "",
        "booked_at": "",
        "queue_status": "No Appointment",
        "has_live_appointment": False,
        "age": "",
        "gender": "",
        "symptoms": "",
        "symptom_count": "",
        "glucose": "",
        "blood_pressure_systolic": "",
        "blood_pressure_diastolic": "",
        "height_cm": "",
        "weight_kg": "",
        "calculated_bmi": "",
        "smoking_habit": "",
        "alcohol_habit": "",
        "average_sleep_hours": "",
        "family_history": "",
        "medical_history": "",
        "nurse_updated_by": "",
        "nurse_notes": "",
        "doctor_reviewed_by": "",
        "reviewed_at": "",
        "updated_at": "",
    }


def _nurse_patient_records() -> list[Dict[str, Any]]:
    patient_name_lookup = {
        str(row.get("patient_id", "")).strip(): str(row.get("name", "")).strip()
        for row in _load_patients_df().to_dict(orient="records")
        if str(row.get("patient_id", "")).strip()
    }
    profile_lookup = {
        _normalize_patient_id(row.get("patient_id")): dict(row)
        for row in patient_db.list_profiles()
        if _normalize_patient_id(row.get("patient_id"))
    }
    now = datetime.now()
    appointment_lookup: Dict[str, Dict[str, Any]] = {}
    for appointment in patient_db.list_appointments():
        patient_id = _normalize_patient_id(appointment.get("patient_id"))
        if not patient_id:
            continue
        appointment_time = _parse_appointment_time(appointment.get("appointment_time"))
        booked_at = _parse_booked_at(appointment.get("booked_at"))
        is_upcoming = _is_upcoming_appointment_date(appointment_time, reference=now)
        current = appointment_lookup.get(patient_id)
        current_time = _parse_appointment_time((current or {}).get("appointment_time"))
        current_booked = _parse_booked_at((current or {}).get("booked_at"))
        current_upcoming = _is_upcoming_appointment_date(current_time, reference=now)

        should_replace = current is None
        if not should_replace and is_upcoming != current_upcoming:
            should_replace = is_upcoming and not current_upcoming
        if not should_replace and is_upcoming and current_upcoming:
            should_replace = _datetime_sort_value(booked_at) < _datetime_sort_value(current_booked)
        if not should_replace and not is_upcoming and not current_upcoming:
            should_replace = _datetime_sort_value(booked_at) < _datetime_sort_value(current_booked)

        if should_replace:
            appointment_lookup[patient_id] = dict(appointment)

    patient_ids = sorted(
        set(patient_name_lookup.keys()) | set(profile_lookup.keys()) | set(appointment_lookup.keys()),
        key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
    )

    records: list[Dict[str, Any]] = []
    for patient_id in patient_ids:
        profile = dict(profile_lookup.get(patient_id) or _empty_nurse_record(patient_id))
        appointment = appointment_lookup.get(patient_id) or {}
        appointment_time = _parse_appointment_time(appointment.get("appointment_time"))
        booked_at = _parse_booked_at(appointment.get("booked_at"))
        has_live_appointment = _is_upcoming_appointment_date(appointment_time, reference=now)
        profile["patient_id"] = patient_id
        if not str(profile.get("patient_name", "")).strip():
            profile["patient_name"] = patient_name_lookup.get(patient_id, "")
        profile["appointment_id"] = str(appointment.get("appointment_id", "")).strip()
        profile["appointment_time"] = appointment_time.isoformat() if appointment_time else str(appointment.get("appointment_time", "")).strip()
        profile["booked_at"] = booked_at.isoformat() if booked_at else str(appointment.get("booked_at", "")).strip()
        profile["has_live_appointment"] = has_live_appointment
        profile["queue_status"] = "Live Appointment" if has_live_appointment else ("Previous Booking" if appointment else "No Appointment")
        records.append(profile)
    records.sort(
        key=lambda item: (
            0 if item.get("has_live_appointment") else (1 if item.get("appointment_id") else 2),
            _datetime_sort_value(_parse_booked_at(item.get("booked_at"))),
            _datetime_sort_value(_parse_appointment_time(item.get("appointment_time"))),
            (0, int(str(item.get("patient_id")))) if str(item.get("patient_id")).isdigit() else (1, str(item.get("patient_id"))),
        )
    )
    return records


def _nurse_record_payload(form_data: Dict[str, Any], nurse_id: str) -> Dict[str, Any]:
    patient_id = _normalize_patient_id(form_data.get("patient_id"))
    age = _to_int(str(form_data.get("age", "")), "Age", 0, 130)
    symptom_count = _to_int(str(form_data.get("symptom_count", "")), "Symptom Count", 0, 100)
    glucose = _to_float(str(form_data.get("glucose", "")), "Glucose", 0, 1000)
    systolic = _to_float(str(form_data.get("blood_pressure_systolic", "")), "Blood Pressure Systolic", 0, 300)
    diastolic = _to_float(str(form_data.get("blood_pressure_diastolic", "")), "Blood Pressure Diastolic", 0, 250)
    height_cm = _to_float(str(form_data.get("height_cm", "")), "Height (cm)", 1, 300)
    weight_kg = _to_float(str(form_data.get("weight_kg", "")), "Weight (kg)", 1, 500)
    average_sleep_hours = _normalize_average_sleep_hours(form_data.get("average_sleep_hours"), required=True)
    bmi = _calculate_bmi(height_cm, weight_kg)

    gender = str(form_data.get("gender", "")).strip().lower()
    smoking_habit = str(form_data.get("smoking_habit", "")).strip().lower()
    alcohol_habit = str(form_data.get("alcohol_habit", "")).strip().lower()
    family_history = str(form_data.get("family_history", "")).strip().lower()
    symptoms = _sanitize_symptoms_text(form_data.get("symptoms", ""))
    if gender not in {"male", "female", "other"}:
        raise ValueError("Gender must be one of: male, female, other.")
    if smoking_habit not in {"yes", "no"}:
        raise ValueError("Smoking Habit must be yes or no.")
    if alcohol_habit not in {"yes", "no"}:
        raise ValueError("Alcohol Habit must be yes or no.")
    if family_history not in {"yes", "no"}:
        raise ValueError("Family History must be yes or no.")
    if not symptoms:
        raise ValueError("Symptoms are required.")

    blood_pressure_text = f"{int(round(systolic))}/{int(round(diastolic))}"
    patient_name = _get_patient_name_by_id(patient_id)
    reviewed_by = _safe_db_text(form_data.get("doctor_reviewed_by"))
    reviewed_at = _safe_db_text(form_data.get("reviewed_at"))
    return {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "Age": age,
        "Gender": gender,
        "Symptoms": symptoms,
        "Symptom_Count": symptom_count,
        "Glucose": glucose,
        "BloodPressure": blood_pressure_text,
        "BMI": round(bmi, 2),
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "BMI_Category": patient_db.weight_category_from_bmi(bmi),
        "Smoking_Habit": smoking_habit,
        "Alcohol_Habit": alcohol_habit,
        "Average_Sleep_Hours": average_sleep_hours,
        "Medical_History": str(form_data.get("medical_history", "")).strip(),
        "Family_History": family_history,
        "Nurse_Updated_By": nurse_id,
        "Nurse_Notes": str(form_data.get("nurse_notes", "")).strip(),
        "Doctor_Reviewed_By": reviewed_by,
        "Reviewed_At": reviewed_at,
        HEALTH_DATA_SUBMITTED_AT_COLUMN: _utc_now_iso(),
    }


def _nurse_patient_features(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Age": record["Age"],
        "Gender": record["Gender"],
        "Symptoms": record["Symptoms"],
        "Symptom_Count": record["Symptom_Count"],
        "Glucose": record["Glucose"],
        "BloodPressure": record["BloodPressure"],
        "BMI": record["BMI"],
        "Height_cm": record["Height_cm"],
        "Weight_kg": record["Weight_kg"],
        "BMI_Category": record["BMI_Category"],
        "Smoking_Habit": record["Smoking_Habit"],
        "Alcohol_Habit": record["Alcohol_Habit"],
        "Average_Sleep_Hours": record["Average_Sleep_Hours"],
        "Medical_History": record["Medical_History"],
        "Family_History": record["Family_History"],
        "Nurse_Updated_By": record["Nurse_Updated_By"],
        "Nurse_Notes": record["Nurse_Notes"],
        "Doctor_Reviewed_By": record["Doctor_Reviewed_By"],
        "Reviewed_At": record["Reviewed_At"],
        HEALTH_DATA_SUBMITTED_AT_COLUMN: record[HEALTH_DATA_SUBMITTED_AT_COLUMN],
    }


def _build_risk_report(patient_id: str, patient_features: Dict[str, Any]) -> Dict[str, Any]:
    csv_row = _build_csv_row_from_features(patient_id, patient_features)
    normalized_features = _build_features_from_new_patient_row(csv_row)
    result = risk_engine.predict(normalized_features)
    priority = determine_priority(result.risk_level)
    payload = result.model_dump()
    payload.update(
        {
            "appointment_priority": priority.priority,
            "recommended_slot": priority.recommended_slot,
            "priority_badge_text": priority.badge_text,
            "priority_badge_color": priority.badge_color,
        }
    )
    return payload


def _patient_features_from_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    systolic = _to_float_or_none(profile.get("blood_pressure_systolic"))
    diastolic = _to_float_or_none(profile.get("blood_pressure_diastolic"))
    blood_pressure: Any = ""
    if systolic is not None and diastolic is not None:
        blood_pressure = f"{int(round(systolic))}/{int(round(diastolic))}"
    elif systolic is not None:
        blood_pressure = systolic

    bmi = _to_float_or_none(profile.get("calculated_bmi"))
    if bmi is None:
        height_cm = _to_float_or_none(profile.get("height_cm"))
        weight_kg = _to_float_or_none(profile.get("weight_kg"))
        if height_cm is not None and weight_kg is not None and height_cm > 0 and weight_kg > 0:
            bmi = round(_calculate_bmi(height_cm, weight_kg), 2)

    return {
        "Age": _to_float_or_none(profile.get("age")),
        "Gender": str(profile.get("gender", "")).strip().lower(),
        "Symptoms": str(profile.get("symptoms", "")).strip(),
        "Symptom_Count": _to_float_or_none(profile.get("symptom_count")),
        "Glucose": _to_float_or_none(profile.get("glucose")),
        "BloodPressure": blood_pressure,
        "BMI": bmi,
        "Height_cm": _to_float_or_none(profile.get("height_cm")),
        "Weight_kg": _to_float_or_none(profile.get("weight_kg")),
        "BMI_Category": str(profile.get("weight_category", "")).strip() or patient_db.weight_category_from_bmi(bmi),
        "Smoking_Habit": str(profile.get("smoking_habit", "")).strip().lower(),
        "Alcohol_Habit": str(profile.get("alcohol_habit", "")).strip().lower(),
        "Average_Sleep_Hours": _to_float_or_none(profile.get("average_sleep_hours")),
        "Medical_History": str(profile.get("medical_history", "")).strip(),
        "Family_History": str(profile.get("family_history", "")).strip().lower(),
        "Nurse_Updated_By": str(profile.get("nurse_updated_by", "")).strip(),
        "Nurse_Notes": str(profile.get("nurse_notes", "")).strip(),
        "Doctor_Reviewed_By": str(profile.get("doctor_reviewed_by", "")).strip(),
        "Reviewed_At": str(profile.get("reviewed_at", "")).strip(),
        HEALTH_DATA_SUBMITTED_AT_COLUMN: _normalize_submission_timestamp(profile.get("health_data_submitted_at")),
    }


DOCTOR_PROFILE_COLUMNS = ["doctor_id", "doctor_name", "specialization", "available_time", "emergency_doctor"]
DOCTOR_LEAVE_COLUMNS = ["doctor_id", "leave_date"]


def _normalize_emergency_flag(value: Any) -> str:
    return "yes" if str(value or "").strip().lower() in {"1", "true", "yes", "y"} else "no"


def _ensure_aux_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    try:
        df = pd.read_csv(path)
    except Exception:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    changed = False
    for col in columns:
        if col not in df.columns:
            df[col] = ""
            changed = True
    if changed:
        df[columns].to_csv(path, index=False)


def _load_doctor_profiles_df() -> pd.DataFrame:
    _ensure_aux_csv(DOCTOR_PROFILE_CSV, DOCTOR_PROFILE_COLUMNS)
    try:
        profile_df = pd.read_csv(DOCTOR_PROFILE_CSV)
    except Exception:
        return pd.DataFrame(columns=DOCTOR_PROFILE_COLUMNS)

    for col in DOCTOR_PROFILE_COLUMNS:
        if col not in profile_df.columns:
            profile_df[col] = ""

    profile_df = profile_df[DOCTOR_PROFILE_COLUMNS].copy()
    profile_df["doctor_id"] = profile_df["doctor_id"].astype(str).str.strip()
    profile_df["doctor_name"] = profile_df["doctor_name"].astype(str).str.strip()
    profile_df["specialization"] = profile_df["specialization"].astype(str).str.strip()
    profile_df["available_time"] = profile_df["available_time"].astype(str).str.strip()
    profile_df["emergency_doctor"] = profile_df["emergency_doctor"].map(_normalize_emergency_flag)
    profile_df = profile_df[profile_df["doctor_id"] != ""]
    profile_df["_doctor_key"] = profile_df["doctor_id"].map(_normalize_doctor_id)
    profile_df = profile_df.drop_duplicates(subset=["_doctor_key"], keep="first").drop(columns=["_doctor_key"])

    doctor_ids = _load_doctor_ids()
    if doctor_ids:
        existing = {_normalize_doctor_id(v) for v in profile_df["doctor_id"].tolist() if str(v).strip()}
        missing = [did for did in doctor_ids if _normalize_doctor_id(did) not in existing]
        if missing:
            additions = pd.DataFrame(
                [
                    {
                        "doctor_id": did,
                        "doctor_name": did,
                        "specialization": "General Medicine",
                        "available_time": "10:00 AM to 9:00 PM",
                        "emergency_doctor": "no",
                    }
                    for did in missing
                ]
            )
            profile_df = pd.concat([profile_df[DOCTOR_PROFILE_COLUMNS], additions], ignore_index=True)
    profile_df = profile_df[DOCTOR_PROFILE_COLUMNS]
    profile_df["emergency_doctor"] = profile_df["emergency_doctor"].map(_normalize_emergency_flag)
    profile_df.to_csv(DOCTOR_PROFILE_CSV, index=False)
    return profile_df[DOCTOR_PROFILE_COLUMNS]


def _load_doctor_leave_df() -> pd.DataFrame:
    _ensure_aux_csv(DOCTOR_LEAVE_CSV, DOCTOR_LEAVE_COLUMNS)
    try:
        leave_df = pd.read_csv(DOCTOR_LEAVE_CSV)
    except Exception:
        return pd.DataFrame(columns=DOCTOR_LEAVE_COLUMNS)

    for col in DOCTOR_LEAVE_COLUMNS:
        if col not in leave_df.columns:
            leave_df[col] = ""

    leave_df["doctor_id"] = leave_df["doctor_id"].astype(str).str.strip()
    leave_df["leave_date"] = leave_df["leave_date"].astype(str).str.strip().str.slice(0, 10)
    return leave_df[DOCTOR_LEAVE_COLUMNS]


def _doctor_profile_by_id(doctor_id: str) -> Dict[str, str]:
    did = str(doctor_id or "").strip()
    if not did:
        return {}
    normalized_did = _normalize_doctor_id(did)
    profiles = _load_doctor_profiles_df()
    matches = profiles[profiles["doctor_id"].astype(str).map(_normalize_doctor_id) == normalized_did]
    if matches.empty:
        return {
            "doctor_id": did,
            "doctor_name": did,
            "specialization": "General Medicine",
            "available_time": "10:00 AM to 9:00 PM",
            "emergency_doctor": "no",
        }
    row = matches.iloc[0].to_dict()
    return {
        "doctor_id": str(row.get("doctor_id", did)).strip(),
        "doctor_name": str(row.get("doctor_name", did)).strip() or did,
        "specialization": str(row.get("specialization", "General Medicine")).strip() or "General Medicine",
        "available_time": str(row.get("available_time", "10:00 AM to 9:00 PM")).strip() or "10:00 AM to 9:00 PM",
        "emergency_doctor": _normalize_emergency_flag(row.get("emergency_doctor", "no")),
    }


def _is_doctor_on_leave(doctor_id: str, appointment_date: datetime) -> bool:
    did = _normalize_doctor_id(doctor_id)
    if not did:
        return False
    date_key = appointment_date.date().isoformat()
    leaves = _load_doctor_leave_df()
    if leaves.empty:
        return False
    mask = (leaves["doctor_id"].astype(str).map(_normalize_doctor_id) == did) & (
        leaves["leave_date"].astype(str).str.strip() == date_key
    )
    return bool(mask.any())


def _find_alternative_doctor(selected_doctor_id: str, appointment_date: datetime) -> Optional[Dict[str, str]]:
    selected = _doctor_profile_by_id(selected_doctor_id)
    selected_spec = str(selected.get("specialization", "General Medicine")).strip().lower()
    profiles = _load_doctor_profiles_df()
    if profiles.empty:
        return None

    profiles = profiles.copy()
    profiles["doctor_id"] = profiles["doctor_id"].astype(str).str.strip()
    selected_norm = _normalize_doctor_id(selected_doctor_id)
    profiles = profiles[profiles["doctor_id"].map(_normalize_doctor_id) != selected_norm]
    if selected_spec:
        profiles = profiles[
            profiles["specialization"].astype(str).str.strip().str.lower() == selected_spec
        ]

    if profiles.empty:
        return None

    profiles = profiles.sort_values(by=["doctor_name", "doctor_id"], kind="stable")
    for _, row in profiles.iterrows():
        candidate_id = str(row.get("doctor_id", "")).strip()
        if not candidate_id:
            continue
        if _is_doctor_on_leave(candidate_id, appointment_date):
            continue
        return {
            "doctor_id": candidate_id,
            "doctor_name": str(row.get("doctor_name", candidate_id)).strip() or candidate_id,
            "specialization": str(row.get("specialization", "General Medicine")).strip() or "General Medicine",
            "available_time": str(row.get("available_time", "10:00 AM to 9:00 PM")).strip() or "10:00 AM to 9:00 PM",
            "emergency_doctor": _normalize_emergency_flag(row.get("emergency_doctor", "no")),
        }
    return None


def _is_truthy_text(value: Any) -> bool:
    return _normalize_emergency_flag(value) == "yes"


def _is_night_emergency_window(appointment_time: datetime) -> bool:
    # Night window spans 7:00 PM through 6:59 AM (crosses midnight).
    minutes = appointment_time.hour * 60 + appointment_time.minute
    return minutes >= 19 * 60 or minutes < 7 * 60


def _is_emergency_doctor_profile(profile: Dict[str, Any]) -> bool:
    return _is_truthy_text(profile.get("emergency_doctor", ""))


def _available_emergency_doctors(appointment_date: datetime, exclude_doctor_id: str = "") -> list[Dict[str, str]]:
    profiles = _load_doctor_profiles_df()
    if profiles.empty:
        return []
    exclude = _normalize_doctor_id(exclude_doctor_id)
    profiles = profiles.copy()
    profiles["doctor_id"] = profiles["doctor_id"].astype(str).str.strip()
    if exclude:
        profiles = profiles[profiles["doctor_id"].map(_normalize_doctor_id) != exclude]
    profiles = profiles[profiles["emergency_doctor"].map(_is_truthy_text)]
    if profiles.empty:
        return []

    rows: list[Dict[str, str]] = []
    profiles = profiles.sort_values(by=["doctor_name", "doctor_id"], kind="stable")
    for _, row in profiles.iterrows():
        did = str(row.get("doctor_id", "")).strip()
        if not did or _is_doctor_on_leave(did, appointment_date):
            continue
        rows.append(
            {
                "doctor_id": did,
                "doctor_name": str(row.get("doctor_name", did)).strip() or did,
                "specialization": str(row.get("specialization", "General Medicine")).strip() or "General Medicine",
                "available_time": str(row.get("available_time", "10:00 AM to 9:00 PM")).strip() or "10:00 AM to 9:00 PM",
                "emergency_doctor": _normalize_emergency_flag(row.get("emergency_doctor", "no")),
            }
        )
    return rows


def _add_doctor_leave(doctor_id: str, leave_date: str) -> None:
    did = str(doctor_id or "").strip()
    date_text = str(leave_date or "").strip()[:10]
    if not did or not date_text:
        raise ValueError("doctor_id and leave_date are required")

    try:
        parsed = datetime.fromisoformat(date_text)
    except ValueError as exc:
        raise ValueError("leave_date must be in YYYY-MM-DD format") from exc
    date_key = parsed.date().isoformat()

    leaves = _load_doctor_leave_df()
    did_norm = _normalize_doctor_id(did)
    already = (
        (leaves["doctor_id"].astype(str).map(_normalize_doctor_id) == did_norm)
        & (leaves["leave_date"].astype(str).str.strip() == date_key)
    )
    if bool(already.any()):
        return

    updated = pd.concat([leaves, pd.DataFrame([{"doctor_id": did, "leave_date": date_key}])], ignore_index=True)
    updated = updated[DOCTOR_LEAVE_COLUMNS]
    updated.to_csv(DOCTOR_LEAVE_CSV, index=False)


def _remove_doctor_leave(doctor_id: str, leave_date: str) -> None:
    did = str(doctor_id or "").strip()
    date_text = str(leave_date or "").strip()[:10]
    if not did or not date_text:
        raise ValueError("doctor_id and leave_date are required")
    try:
        parsed = datetime.fromisoformat(date_text)
    except ValueError as exc:
        raise ValueError("leave_date must be in YYYY-MM-DD format") from exc
    date_key = parsed.date().isoformat()

    leaves = _load_doctor_leave_df()
    did_norm = _normalize_doctor_id(did)
    filtered = leaves[
        ~(
            (leaves["doctor_id"].astype(str).map(_normalize_doctor_id) == did_norm)
            & (leaves["leave_date"].astype(str).str.strip() == date_key)
        )
    ]
    filtered = filtered[DOCTOR_LEAVE_COLUMNS]
    filtered.to_csv(DOCTOR_LEAVE_CSV, index=False)


def _normalize_patient_id(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _normalize_doctor_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _sanitize_symptoms_text(value: Any) -> str:
    text = str(value or "")
    text = SYMPTOMS_SANITIZE_PATTERN.sub("", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^,\s*|\s*,\s*$", "", text)
    return text.strip()


def _bootstrap_patient_db_from_csv() -> None:
    try:
        ensure_csv_exists(NEW_PATIENT_CSV)
        records_df = pd.read_csv(NEW_PATIENT_CSV)
    except Exception:
        return

    name_lookup: Dict[str, str] = {}
    try:
        patients_df = _load_patients_df()
        for _, rec in patients_df.iterrows():
            pid = _normalize_patient_id(rec.get("patient_id"))
            name_lookup[pid] = str(rec.get("name", "")).strip()
    except Exception:
        name_lookup = {}

    for _, rec in records_df.iterrows():
        row = rec.to_dict()
        pid = _normalize_patient_id(row.get("Patient_ID"))
        _save_patient_profile_to_db(row, name_lookup.get(pid, ""))


_bootstrap_patient_db_from_csv()


def _default_feature_value(feature_name: str) -> Any:
    numeric_defaults = {
        "Age": 45,
        "Symptom_Count": 1,
        "Glucose": 95,
        "BloodPressure": 120,
        "BMI": 24.5,
    }
    categorical_defaults = {
        "Gender": 1,
        "Symptoms": "none",
        "Age_Group": "Adult",
        "BMI_Category": "Normal",
        "BP_Category": "Normal",
    }
    if feature_name in numeric_defaults:
        return numeric_defaults[feature_name]
    if feature_name in categorical_defaults:
        return categorical_defaults[feature_name]
    if feature_name.startswith("SYM_"):
        return 0
    return 0


def _age_group(age: float) -> str:
    if age < 13:
        return "Child"
    if age < 20:
        return "Teen"
    if age < 40:
        return "Adult"
    if age < 60:
        return "Middle_Age"
    return "Senior"


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _bp_category(bp: float) -> str:
    if bp < 80:
        return "Low"
    if bp <= 120:
        return "Normal"
    if bp <= 139:
        return "Elevated"
    return "High"


def _calculate_bmi(height_cm: float, weight_kg: float) -> float:
    if height_cm <= 0:
        raise ValueError("Height (cm) must be greater than 0.")
    if weight_kg <= 0:
        raise ValueError("Weight (kg) must be greater than 0.")
    height_m = height_cm / 100.0
    return weight_kg / (height_m * height_m)


def _build_features_from_new_patient_row(row: Dict[str, Any]) -> Dict[str, Any]:
    expected_cols = list(getattr(risk_engine.model, "feature_names_in_", []))
    if expected_cols:
        features = {col: _default_feature_value(str(col)) for col in expected_cols}
    else:
        features = {}

    age = float(row.get("Age", 45))
    gender_raw = str(row.get("Gender", "male")).strip().lower()
    gender_map = {"male": 1, "female": 0, "other": -1}
    gender = gender_map.get(gender_raw, 1)
    symptoms = str(row.get("Symptoms", "")).strip()
    symptom_count = int(float(row.get("Symptom_Count", 0)))
    glucose = float(row.get("Glucose", 95))
    blood_pressure = float(row.get("BloodPressure", 120))
    bmi = float(row.get("BMI", 24.5))
    height_cm = _to_float_or_none(row.get("Height_cm"))
    weight_kg = _to_float_or_none(row.get("Weight_kg"))
    if height_cm is not None and weight_kg is not None and height_cm > 0 and weight_kg > 0:
        bmi = _calculate_bmi(height_cm, weight_kg)
    smoking_habit = str(row.get("Smoking_Habit", "")).strip().lower()
    alcohol_habit = str(row.get("Alcohol_Habit", "")).strip().lower()
    average_sleep_hours = _to_float_or_none(row.get("Average_Sleep_Hours"))
    medical_history = str(row.get("Medical_History", "")).strip()
    family_history = str(row.get("Family_History", "")).strip().lower()
    health_data_submitted_at = _normalize_submission_timestamp(row.get(HEALTH_DATA_SUBMITTED_AT_COLUMN))

    features.update(
        {
            "Age": age,
            "Gender": gender,
            "Symptoms": symptoms,
            "Symptom_Count": symptom_count,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "BMI": bmi,
            "Height_cm": height_cm,
            "Weight_kg": weight_kg,
            "Smoking_Habit": smoking_habit,
            "Alcohol_Habit": alcohol_habit,
            "Average_Sleep_Hours": average_sleep_hours,
            "Medical_History": medical_history,
            "Family_History": family_history,
            HEALTH_DATA_SUBMITTED_AT_COLUMN: health_data_submitted_at,
            "health_data_submitted_at": health_data_submitted_at,
            "Age_Group": _age_group(age),
            "BMI_Category": _bmi_category(bmi),
            "BP_Category": _bp_category(blood_pressure),
        }
    )

    if symptoms:
        symptom_tokens = [s.strip().lower().replace(" ", "_") for s in symptoms.split(",") if s.strip()]
        for token in symptom_tokens:
            sym_col = f"SYM_{token}"
            if sym_col in features:
                features[sym_col] = 1

    return features


def _load_new_patient_features(patient_id: str) -> Optional[Dict[str, Any]]:
    ensure_csv_exists(NEW_PATIENT_CSV)
    try:
        df = pd.read_csv(NEW_PATIENT_CSV)
    except Exception:
        return None

    if "Patient_ID" not in df.columns:
        return None

    pid = _normalize_patient_id(patient_id)
    matches = df[df["Patient_ID"].astype(str).map(_normalize_patient_id) == pid]
    if matches.empty:
        return None

    latest_row = matches.iloc[-1].to_dict()
    return _build_features_from_new_patient_row(latest_row)


def get_features_for_patient(patient_id: str) -> Dict[str, Any]:
    new_features = _load_new_patient_features(patient_id)
    if new_features is not None:
        return new_features
    raise ValueError("Patient_ID not found in new_patient_data.csv")


def _patient_chat_profile(patient_id: str) -> Dict[str, Any]:
    profile = patient_db.get_profile(patient_id)
    return profile if isinstance(profile, dict) else {}


def _patient_chat_appointments(patient_id: str) -> list[Dict[str, Any]]:
    normalized_pid = _normalize_patient_id(patient_id)
    items: list[Dict[str, Any]] = []
    for item in patient_db.list_appointments(patient_id=patient_id):
        if _normalize_patient_id(item.get("patient_id")) != normalized_pid:
            continue
        dt = _parse_appointment_time(item.get("appointment_time"))
        normalized = dict(item)
        normalized["appointment_dt"] = dt
        items.append(normalized)
    items.sort(key=lambda ap: ap.get("appointment_dt") or datetime.min, reverse=True)
    return items


def _chat_text(value: Any, fallback: str = "not available") -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return fallback
    return text


def _chat_number(value: Any, suffix: str = "") -> str:
    if value is None:
        return "not available"
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return _chat_text(value)
    if parsed.is_integer():
        text = str(int(parsed))
    else:
        text = f"{parsed:.1f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def _patient_precautions(profile: Dict[str, Any], features: Dict[str, Any], risk_result: Any) -> list[str]:
    precautions: list[str] = []
    risk_level = _normalize_risk_label(getattr(risk_result, "risk_level", ""))
    glucose = _to_float_or_none(profile.get("glucose"))
    systolic = _to_float_or_none(profile.get("blood_pressure_systolic"))
    bmi = _to_float_or_none(profile.get("calculated_bmi"))
    symptoms = _chat_text(profile.get("symptoms"), "").lower()
    smoking = _chat_text(profile.get("smoking_habit"), "").lower()
    alcohol = _chat_text(profile.get("alcohol_habit"), "").lower()
    sleep_hours = _to_float_or_none(profile.get("average_sleep_hours"))
    sleep_category, sleep_meaning = _sleep_category_details(sleep_hours)

    if risk_level == "High Risk":
        precautions.append("Your current record looks high risk. Follow the recommended urgent consultation slot and do not delay care.")
    elif risk_level == "Medium Risk":
        precautions.append("Your record suggests medium risk. Try to see the doctor on the same day and keep symptoms under observation.")
    else:
        precautions.append("Your record looks lower risk right now, but continue monitoring symptoms and attend routine follow-up.")

    if glucose is not None and glucose >= 126:
        precautions.append("Your glucose appears elevated. Avoid excess sugar, stay hydrated, and discuss diabetic screening with your doctor.")
    if systolic is not None and systolic >= 140:
        precautions.append("Your blood pressure reading is elevated. Reduce salt, rest well, and ask the doctor for a blood pressure review.")
    if bmi is not None and bmi >= 30:
        precautions.append("Your BMI is in the obese range. Ask for a doctor-guided weight, diet, and activity plan.")
    elif bmi is not None and bmi < 18.5:
        precautions.append("Your BMI is in the underweight range. Ask the doctor about nutrition support and underlying causes.")
    if smoking == "yes":
        precautions.append("Smoking increases health risk. The doctor will likely advise stopping smoking and avoiding smoke exposure.")
    if alcohol == "yes":
        precautions.append("Limit alcohol intake and mention your current drinking habit during consultation.")
    if sleep_category == "Poor Sleep":
        precautions.append("Your sleep pattern falls under Poor Sleep. This is a higher-risk sleep pattern and should be discussed with the doctor.")
    elif sleep_category == "Low Sleep":
        precautions.append("Your sleep pattern falls under Low Sleep. You may not be getting enough rest, so discuss sleep quality and routine.")
    elif sleep_category == "Excess Sleep":
        precautions.append("Your sleep pattern falls under Excess Sleep. Mention fatigue, weakness, or daytime drowsiness to the doctor.")
    if "chest pain" in symptoms or "breath" in symptoms:
        precautions.append("Chest pain or breathing-related symptoms should be discussed urgently, especially if they worsen suddenly.")

    medical_history = _chat_text(profile.get("medical_history"), "")
    if medical_history:
        precautions.append(f"Share your medical history clearly with the doctor: {medical_history}.")

    deduped: list[str] = []
    for item in precautions:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _doctor_match_reason(symptoms_text: str, risk_label: str) -> str:
    symptoms_text = symptoms_text.lower()
    if "chest pain" in symptoms_text or "breath" in symptoms_text:
        return "Your symptoms should be reviewed promptly because chest pain or breathing issues may need urgent evaluation."
    if "fever" in symptoms_text or "cough" in symptoms_text:
        return "Start with a general medicine doctor for symptom review and next-step treatment."
    if "headache" in symptoms_text or "dizziness" in symptoms_text:
        return "A general medicine doctor is a good first consultation for headache, dizziness, and vital-sign review."
    if risk_label == "High Risk":
        return "Because your AI risk is high, emergency-capable doctors are ranked first."
    if risk_label == "Medium Risk":
        return "Because your AI risk is medium, faster same-day review is preferred."
    return "A general medicine doctor is the best first consultation based on the current data in your portal."


def _recommended_doctors_for_patient(symptoms_text: str, risk_label: str) -> list[Dict[str, str]]:
    profiles = _load_doctor_profiles_df()
    if profiles.empty:
        return []

    rows: list[Dict[str, str]] = []
    prioritize_emergency = risk_label == "High Risk"
    profiles = profiles.sort_values(by=["doctor_name", "doctor_id"], kind="stable")
    for _, row in profiles.iterrows():
        doctor = {
            "doctor_id": str(row.get("doctor_id", "")).strip(),
            "doctor_name": str(row.get("doctor_name", "")).strip() or str(row.get("doctor_id", "")).strip(),
            "specialization": str(row.get("specialization", "General Medicine")).strip() or "General Medicine",
            "available_time": str(row.get("available_time", "10:00 AM to 9:00 PM")).strip() or "10:00 AM to 9:00 PM",
            "emergency_doctor": _normalize_emergency_flag(row.get("emergency_doctor", "no")),
        }
        if doctor["doctor_id"]:
            rows.append(doctor)

    rows.sort(
        key=lambda item: (
            0 if (prioritize_emergency and item.get("emergency_doctor") == "yes") else 1,
            0 if item.get("emergency_doctor") == "yes" else 1,
            item.get("doctor_name", "").lower(),
            item.get("doctor_id", "").lower(),
        )
    )
    return rows[:4]


def _get_openai_chat_client() -> Any:
    global _openai_chat_client, _openai_chat_client_ready
    if _openai_chat_client_ready:
        return _openai_chat_client

    _openai_chat_client_ready = True
    if not OPENAI_API_KEY:
        return None

    try:
        from openai import OpenAI # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        app.logger.warning("OpenAI SDK unavailable for patient chat: %s", exc)
        return None

    try:
        _openai_chat_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:  # pragma: no cover - startup guard
        app.logger.warning("OpenAI client initialization failed: %s", exc)
        _openai_chat_client = None
    return _openai_chat_client


def _extract_openai_output_text(response: Any) -> str:
    text = str(getattr(response, "output_text", "") or "").strip()
    if text:
        return text

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            candidate = str(getattr(content, "text", "") or "").strip()
            if candidate:
                return candidate
    return ""


def _patient_chat_openai_reply(
    *,
    user_message: str,
    context: Dict[str, Any],
) -> Optional[str]:
    client = _get_openai_chat_client()
    if client is None:
        return None

    system_prompt = (
        "You are an AI assistant inside a healthcare portal. "
        "If the patient asks about their own record, appointments, doctors, risk level, or precautions, answer only from the supplied patient context. "
        "If the patient asks a general question that is not about their portal data, answer it like a normal helpful AI assistant. "
        "Never invent patient-specific appointments, doctor names, medical facts, or measurements that are not present in the supplied context. "
        "If a medical question requires diagnosis or urgent care, give a brief safety-minded answer and remind the user to contact a clinician or emergency services when appropriate. "
        "Keep the answer concise, practical, and easy to understand."
    )
    user_prompt = (
        "Patient portal context:\n"
        f"{json.dumps(context, ensure_ascii=True, indent=2)}\n\n"
        f"Patient question:\n{user_message}\n"
    )

    try:
        response = client.responses.create(
            model=OPENAI_CHAT_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=300,
        )
    except Exception as exc:  # pragma: no cover - network/service dependent
        app.logger.warning("OpenAI patient chat request failed: %s", exc)
        return None

    reply = _extract_openai_output_text(response)
    return reply or None


def _is_general_chat_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False

    portal_keywords = (
        "my profile",
        "my details",
        "patient details",
        "patient detail",
        "show my profile",
        "show my details",
        "my medical history",
        "my family history",
        "my blood pressure",
        "my bp",
        "my glucose",
        "my sugar",
        "my bmi",
        "my weight",
        "my height",
        "my vitals",
        "my risk level",
        "risk summary",
        "current risk",
        "my appointment",
        "next appointment",
        "doctor assigned",
        "appointment",
        "booking",
        "patient portal",
        "record",
        "portal",
    )
    return not any(keyword in normalized for keyword in portal_keywords)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = str(text or "").strip().lower()
    return any(phrase in normalized for phrase in phrases)


def _is_portal_profile_question(message: str) -> bool:
    return _contains_any(
        message,
        (
            "my profile",
            "my details",
            "patient details",
            "patient detail",
            "show my details",
            "show my profile",
            "my age",
            "my gender",
            "my patient",
            "my record",
            "my portal",
            "name age gender",
        ),
    )


def _is_portal_history_question(message: str) -> bool:
    return _contains_any(
        message,
        (
            "my symptoms",
            "my symptom",
            "medical history",
            "family history",
            "smoking habit",
            "alcohol habit",
            "my history",
            "health history",
            "in my record",
            "on my record",
        ),
    )


def _is_portal_vitals_question(message: str) -> bool:
    return _contains_any(
        message,
        (
            "my bp",
            "my blood pressure",
            "my glucose",
            "my sugar",
            "my bmi",
            "my weight",
            "my height",
            "my vitals",
            "vitals summary",
            "show my vitals",
        ),
    )


def _is_portal_risk_question(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if "risk" not in normalized and "prediction" not in normalized:
        return False
    return _contains_any(
        normalized,
        (
            "my risk",
            "risk level",
            "risk summary",
            "my prediction",
            "ai risk",
            "current risk",
            "portal risk",
        ),
    )


def _is_portal_appointment_question(message: str) -> bool:
    return _contains_any(
        message,
        (
            "my appointment",
            "next appointment",
            "latest appointment",
            "doctor assigned",
            "doctor name on appointment",
            "booking confirmation",
            "appointment with",
            "when is my appointment",
            "who is my next doctor",
        ),
    )


def _is_portal_doctor_list_question(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if not (("doctor" in normalized) or ("doctors" in normalized)):
        return False
    return _contains_any(
        normalized,
        (
            "for my condition",
            "for me",
            "recommended doctor",
            "suggested doctor",
            "doctor list",
            "available doctor",
            "show doctors",
            "consult for my",
            "who should i consult for my",
        ),
    )


def _is_portal_precaution_question(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if "precaution" in normalized:
        return True
    return _contains_any(
        normalized,
        (
            "doctor advice for me",
            "advice for my condition",
            "based on my record",
            "my care plan",
            "my treatment plan",
            "what should i do based on my record",
        ),
    )


def _is_general_greeting(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    if not normalized:
        return False
    if re.match(r"^(hi|hello|hey)\b", normalized):
        return True
    return any(normalized.startswith(pattern) for pattern in CHAT_GENERAL_PATTERNS)


def _evaluate_safe_math(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Num):  # pragma: no cover - compatibility
        return float(node.n)
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_MATH_OPERATORS:
        left = _evaluate_safe_math(node.left)
        right = _evaluate_safe_math(node.right)
        return float(SAFE_MATH_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_MATH_OPERATORS:
        operand = _evaluate_safe_math(node.operand)
        return float(SAFE_MATH_OPERATORS[type(node.op)](operand))
    raise ValueError("Unsupported math expression")


def _try_math_answer(message: str) -> Optional[str]:
    text = str(message or "").strip()
    if not text:
        return None

    candidate = re.sub(r"^(what is|calculate|solve)\s+", "", text, flags=re.IGNORECASE).strip(" ?=")
    if not candidate or len(candidate) > 80:
        return None
    if not re.fullmatch(r"[\d\s+\-*/%().]+", candidate):
        return None

    try:
        parsed = ast.parse(candidate, mode="eval")
        result = _evaluate_safe_math(parsed.body)
    except Exception:
        return None

    if result.is_integer():
        return f"The answer is {int(result)}."
    return f"The answer is {result:.4f}".rstrip("0").rstrip(".") + "."


def _to_float_or_none(value: Any) -> Optional[float]:
    try:
        text = str(value or "").strip()
        return float(text) if text else None
    except Exception:
        return None


def _doctor_or_hospital_suggestion(user_message: str, context: Dict[str, Any]) -> Optional[str]:
    normalized = str(user_message or "").strip().lower()
    profile = context.get("profile", {}) if isinstance(context, dict) else {}
    risk_summary = context.get("risk_summary", {}) if isinstance(context, dict) else {}
    bp_sys = _to_float_or_none(profile.get("blood_pressure_systolic"))
    bp_dia = _to_float_or_none(profile.get("blood_pressure_diastolic"))
    risk_label = _chat_text(risk_summary.get("risk_level"), "").lower()

    health_terms = (
        "salt",
        "salty",
        "sodium",
        "food",
        "diet",
        "eat",
        "nutrition",
        "doctor",
        "specialist",
        "hospital",
        "fever",
        "cough",
        "cold",
        "sore throat",
        "headache",
        "migraine",
        "stomach",
        "abdomen",
        "vomit",
        "vomiting",
        "diarrhea",
        "loose motion",
        "chest pain",
        "shortness of breath",
        "breathing trouble",
        "sleep",
        "insomnia",
        "exercise",
        "workout",
        "walking",
        "fitness",
        "dizziness",
        "weakness",
        "faint",
        "bp",
        "blood pressure",
        "glucose",
        "sugar",
        "symptom",
        "signs of",
        "pain",
        "rash",
        "infection",
    )
    if not any(term in normalized for term in health_terms) and "high risk" not in risk_label:
        return None

    emergency_terms = (
        "chest pain",
        "shortness of breath",
        "breathing trouble",
        "can't breathe",
        "seizure",
        "faint",
        "unconscious",
        "stroke",
        "paralysis",
        "blood in vomit",
        "blood in stool",
        "severe bleeding",
    )
    urgent_terms = (
        "high fever",
        "severe headache",
        "vomiting",
        "diarrhea",
        "stomach pain",
        "dizziness",
        "palpitation",
        "weakness",
    )

    if any(term in normalized for term in emergency_terms):
        return "Suggestion: go to the hospital or emergency care immediately."
    if "high risk" in risk_label:
        return "Suggestion: because your portal shows high risk, go to the hospital or see a doctor as soon as possible."
    if bp_sys is not None and bp_dia is not None:
        if bp_sys >= 180 or bp_dia >= 120:
            return "Suggestion: your blood pressure is dangerously high, so go to the hospital immediately."
        if bp_sys < 90 or bp_dia < 60:
            return "Suggestion: if low blood pressure comes with dizziness, fainting, or weakness, go to a hospital or urgent clinic promptly."
    if any(term in normalized for term in urgent_terms):
        return "Suggestion: book a doctor visit soon, and go to the hospital if symptoms become severe or sudden."
    if any(term in normalized for term in ("fever", "cough", "cold", "sore throat", "headache", "diet", "food", "sleep", "exercise")):
        return "Suggestion: start with rest and home care, but visit a doctor if symptoms do not improve or get worse."
    return "Suggestion: if this problem is continuing, book a doctor appointment for proper evaluation."


def _append_action_suggestion(answer: str, user_message: str, context: Dict[str, Any]) -> str:
    suggestion = _doctor_or_hospital_suggestion(user_message, context)
    if not suggestion:
        return answer
    answer_text = str(answer or "").strip()
    if not answer_text:
        return suggestion
    if suggestion.lower() in answer_text.lower():
        return answer_text
    return f"{answer_text}\n\n{suggestion}"


def _is_health_related_message(message: str) -> bool:
    normalized = str(message or "").strip().lower()
    health_terms = (
        "salt",
        "salty",
        "sodium",
        "food",
        "diet",
        "eat",
        "nutrition",
        "doctor",
        "specialist",
        "hospital",
        "fever",
        "cough",
        "cold",
        "sore throat",
        "headache",
        "migraine",
        "stomach",
        "abdomen",
        "vomit",
        "vomiting",
        "diarrhea",
        "loose motion",
        "chest pain",
        "shortness of breath",
        "breathing trouble",
        "sleep",
        "insomnia",
        "exercise",
        "workout",
        "walking",
        "fitness",
        "dizziness",
        "weakness",
        "faint",
        "bp",
        "blood pressure",
        "glucose",
        "sugar",
        "symptom",
        "signs of",
        "pain",
        "rash",
        "infection",
        "risk",
    )
    return any(term in normalized for term in health_terms)


def _offline_general_chat_reply(user_message: str, context: Dict[str, Any]) -> str:
    normalized = str(user_message or "").strip().lower()
    profile = context.get("profile", {}) if isinstance(context, dict) else {}
    risk_summary = context.get("risk_summary", {}) if isinstance(context, dict) else {}
    risk_label = _chat_text(risk_summary.get("risk_level"), "your current risk level")
    symptoms = _chat_text(profile.get("symptoms"), "not available")
    bp_sys = _chat_text(profile.get("blood_pressure_systolic"), "")
    bp_dia = _chat_text(profile.get("blood_pressure_diastolic"), "")
    bp_sys_num = _to_float_or_none(bp_sys)
    bp_dia_num = _to_float_or_none(bp_dia)

    if _is_general_greeting(normalized):
        return _append_action_suggestion((
            "Hello! Ask me any health or general question and I will do my best to answer here. "
            "I can also help with your patient profile, appointments, risk summary, and doctor precautions."
        ), user_message, context)

    math_answer = _try_math_answer(user_message)
    if math_answer:
        return math_answer

    if "time" in normalized and "table" not in normalized:
        return f"Right now the time is {datetime.now().strftime('%I:%M %p')}."
    if "date" in normalized or "day today" in normalized or "today" == normalized:
        return f"Today is {datetime.now().strftime('%d %B %Y')}."

    if "who are you" in normalized:
        return "I am your AI chatbot inside this portal. You can ask me health questions, simple general questions, and patient-record questions."
    if "what can you do" in normalized:
        return "I can answer health questions, explain basic topics, help with simple calculations, and guide you through your patient profile, appointments, and risk summary."

    if any(term in normalized for term in ("salt", "salty", "sodium")):
        if bp_sys_num is not None and bp_dia_num is not None and (bp_sys_num <= 90 or bp_dia_num <= 60):
            return _append_action_suggestion((
                f"Because your portal blood pressure reading is {bp_sys}/{bp_dia} mmHg, a small to moderate amount of salt may be acceptable if your doctor has advised it for low blood pressure. "
                "Do not overdo salty packaged food. Prefer balanced meals, fluids, and medical advice if you feel dizziness, faintness, or weakness."
            ), user_message, context)
        if bp_sys_num is not None and bp_dia_num is not None and (bp_sys_num >= 140 or bp_dia_num >= 90):
            return _append_action_suggestion((
                f"Because your portal blood pressure reading is {bp_sys}/{bp_dia} mmHg, limiting salty foods is the safer choice. "
                "Choose fresh food, avoid packaged snacks, chips, pickles, and instant foods, and drink enough water."
            ), user_message, context)
        return _append_action_suggestion((
            "Salty foods are best kept in moderation. Too much salt can raise blood pressure, while very low blood pressure may sometimes need a different plan. "
            "If you have dizziness, swelling, kidney problems, or hypertension, follow a doctor's advice."
        ), user_message, context)

    if any(term in normalized for term in ("food", "diet", "eat", "meal", "nutrition")):
        return _append_action_suggestion((
            f"A balanced diet is usually the safest choice: more vegetables, fruits, pulses, whole grains, enough water, and less fried, sugary, and very salty food. "
            f"Based on your portal, your current risk summary is {risk_label.lower()} and your recorded symptoms are {symptoms}."
        ), user_message, context)

    if any(term in normalized for term in ("which doctor", "what doctor", "what type of doctor", "which specialist", "what specialist")):
        if any(term in normalized for term in ("fever", "cold", "cough", "sore throat", "infection")):
            return _append_action_suggestion("A general medicine doctor is usually the best first choice for fever, cough, cold, or sore throat. If breathing becomes difficult, seek urgent care.", user_message, context)
        if any(term in normalized for term in ("skin", "rash", "itching", "acne")):
            return _append_action_suggestion("A dermatologist is usually the right doctor for skin, rash, itching, or acne problems.", user_message, context)
        if any(term in normalized for term in ("headache", "dizziness", "seizure", "numbness")):
            return _append_action_suggestion("A general medicine doctor is a good first step, and a neurologist may be needed if symptoms are severe, repeated, or linked to weakness, seizures, or numbness.", user_message, context)
        if any(term in normalized for term in ("stomach", "abdomen", "vomit", "diarrhea", "acidity")):
            return _append_action_suggestion("A general medicine doctor is a good first choice for stomach problems. A gastroenterologist may be needed if symptoms are ongoing or severe.", user_message, context)
        if any(term in normalized for term in ("chest pain", "heart", "palpitation")):
            return _append_action_suggestion("Chest pain can be serious. Please seek urgent or emergency medical care first, and a cardiology review may be needed.", user_message, context)
        return _append_action_suggestion("A general medicine doctor is usually the best first doctor to consult. They can examine the problem and refer you to a specialist if needed.", user_message, context)

    if any(term in normalized for term in ("fever", "temperature")):
        return _append_action_suggestion((
            "For fever, rest, drink plenty of fluids, eat light food, and monitor the temperature. "
            "Get medical help quickly if the fever is very high, lasts more than a few days, or comes with breathing trouble, confusion, severe weakness, or dehydration."
        ), user_message, context)

    if any(term in normalized for term in ("cough", "cold", "sore throat", "runny nose")):
        return _append_action_suggestion((
            "For a mild cough or cold, rest, warm fluids, steam inhalation, and staying hydrated can help. "
            "Please seek medical care if you have shortness of breath, chest pain, high fever, or symptoms that keep getting worse."
        ), user_message, context)

    if any(term in normalized for term in ("headache", "migraine")):
        return _append_action_suggestion((
            "A mild headache often improves with water, food, rest, and sleep. "
            "Please get urgent care if it is sudden and severe or comes with weakness, confusion, vomiting, fainting, vision changes, or high fever."
        ), user_message, context)

    if any(term in normalized for term in ("stomach pain", "abdomen", "vomit", "vomiting", "diarrhea", "loose motion")):
        return _append_action_suggestion((
            "For stomach upset, focus on fluids, simple food, and rest. "
            "Please get medical help if you have blood in vomit or stool, severe pain, dehydration, or symptoms lasting more than a day or two."
        ), user_message, context)

    if any(term in normalized for term in ("chest pain", "shortness of breath", "breathing trouble", "can't breathe")):
        return _append_action_suggestion("Chest pain or trouble breathing can be serious. Please seek emergency medical care immediately.", user_message, context)

    if any(term in normalized for term in ("sleep", "insomnia", "can't sleep")):
        return _append_action_suggestion((
            "Try a fixed sleep time, less screen use before bed, less caffeine late in the day, and a quiet dark room. "
            "If poor sleep continues for weeks or affects daytime functioning, speak with a doctor."
        ), user_message, context)

    if any(term in normalized for term in ("exercise", "workout", "walking", "fitness")):
        return _append_action_suggestion((
            "For most people, starting with light walking and gradually increasing activity is a good approach. "
            "Stop and seek medical advice if exercise causes chest pain, dizziness, severe breathlessness, or unusual weakness."
        ), user_message, context)

    if "python" in normalized:
        return "Python is a popular programming language known for simple syntax. People use it for web apps, automation, data science, AI, and scripting."
    if "artificial intelligence" in normalized or re.search(r"\bai\b", normalized):
        return "Artificial intelligence is the use of computer systems to perform tasks that usually need human-like reasoning, such as understanding language, finding patterns, and making predictions."
    if "html" in normalized:
        return "HTML is the standard markup language used to structure web pages."
    if "css" in normalized:
        return "CSS is used to style web pages by controlling colors, layout, spacing, fonts, and visual appearance."
    if "javascript" in normalized:
        return "JavaScript is a programming language used to add interaction and dynamic behavior to websites."

    symptoms_of_map = {
        "flu": "Common flu symptoms include fever, cough, sore throat, body aches, tiredness, headache, and sometimes a runny or blocked nose.",
        "common cold": "Common cold symptoms often include sneezing, runny nose, blocked nose, sore throat, mild cough, and sometimes a low fever.",
        "dengue": "Common dengue symptoms include high fever, severe body pain, headache, pain behind the eyes, nausea, rash, and unusual weakness.",
        "malaria": "Common malaria symptoms include fever with chills, sweating, headache, body pain, weakness, nausea, and sometimes vomiting.",
        "diabetes": "Common diabetes symptoms include increased thirst, frequent urination, tiredness, blurry vision, slow wound healing, and unexplained weight change.",
    }
    for condition, answer in symptoms_of_map.items():
        if f"symptoms of {condition}" in normalized or f"signs of {condition}" in normalized:
            return _append_action_suggestion(answer, user_message, context)

    what_is_match = re.match(r"^what is ([a-z0-9 ._-]+)\??$", normalized)
    if what_is_match:
        topic = what_is_match.group(1).strip()
        return f"{topic.title()} is a topic I can help explain. Ask me in a slightly more specific way, like what it is used for, how it works, or why it matters."

    if normalized.endswith("?"):
        return (
            "Here is the best quick answer I can give offline: the right answer depends on the exact context, so the safest next step is to focus on the main goal, avoid risky extremes, and choose the simplest clear option first. "
            "If you make the question a little more specific, I will answer it more directly."
        )

    return (
        "I can help with health guidance, basic explanations, simple calculations, and your patient portal details. "
        "Ask your question in one clear sentence and I will answer as directly as possible."
    )


def _patient_chat_response(patient_id: str, user_message: str) -> Dict[str, Any]:
    profile = _patient_chat_profile(patient_id)
    appointments = _patient_chat_appointments(patient_id)
    try:
        features = get_features_for_patient(patient_id)
    except ValueError:
        features = {}

    risk_result = None
    if features:
        try:
            risk_result = risk_engine.predict(features)
        except Exception:
            risk_result = None

    precautions = _patient_precautions(profile, features, risk_result)
    name = _chat_text(profile.get("patient_name") or session.get("patient_name"), "Patient")
    age = _chat_number(profile.get("age"))
    gender = _chat_text(profile.get("gender"))
    symptoms = _chat_text(profile.get("symptoms"))
    glucose = _chat_number(profile.get("glucose"), " mg/dL")
    bmi = _chat_number(profile.get("calculated_bmi"))
    bp_sys = _chat_number(profile.get("blood_pressure_systolic"))
    bp_dia = _chat_number(profile.get("blood_pressure_diastolic"))
    sleep_hours = _chat_number(profile.get("average_sleep_hours"), " hours")
    sleep_category, sleep_meaning = _sleep_category_details(profile.get("average_sleep_hours"))
    risk_label = _normalize_risk_label(getattr(risk_result, "risk_level", ""))
    recommended_slot = getattr(determine_priority(getattr(risk_result, "risk_level", "")), "recommended_slot", "Next Available Date")
    doctor_reason = _doctor_match_reason(symptoms, risk_label)
    suggested_doctors = _recommended_doctors_for_patient(symptoms, risk_label)
    latest_appointment = appointments[0] if appointments else {}
    future_appointments = [ap for ap in appointments if ap.get("appointment_dt") and ap["appointment_dt"] >= datetime.now()]
    upcoming_appointment = min(future_appointments, key=lambda ap: ap["appointment_dt"]) if future_appointments else {}

    message = str(user_message or "").strip().lower()
    is_general_chat = _is_general_chat_message(message)
    is_profile_question = _is_portal_profile_question(message)
    is_history_question = _is_portal_history_question(message)
    is_vitals_question = _is_portal_vitals_question(message)
    is_risk_question = _is_portal_risk_question(message)
    is_appointment_question = _is_portal_appointment_question(message)
    doctor_list_requested = _is_portal_doctor_list_question(message)
    is_precaution_question = _is_portal_precaution_question(message)

    if _is_general_greeting(message):
        reply = (
            "Hello! You can ask me about your patient profile, appointments, risk summary, and doctor precautions. "
            "If OpenAI is configured, I can also answer general questions here in the same chat."
        )
    elif is_profile_question:
        sleep_text = f" Average sleep: {sleep_hours}."
        if sleep_category:
            sleep_text += f" Sleep category: {sleep_category} ({sleep_meaning})."
        reply = f"Patient summary: {name}, age {age}, gender {gender}. Symptoms: {symptoms}.{sleep_text}"
    elif is_history_question:
        reply = (
            f"Symptoms: {symptoms}. Medical history: {_chat_text(profile.get('medical_history'))}. "
            f"Family history: {_chat_text(profile.get('family_history'))}. Smoking: {_chat_text(profile.get('smoking_habit'))}. "
            f"Alcohol: {_chat_text(profile.get('alcohol_habit'))}. Average sleep: {sleep_hours}"
            + (f" ({sleep_category}: {sleep_meaning})." if sleep_category else ".")
        )
    elif is_vitals_question:
        reply = (
            f"Vitals summary: blood pressure {bp_sys}/{bp_dia} mmHg, glucose {glucose}, BMI {bmi}, "
            f"weight {_chat_number(profile.get('weight_kg'), ' kg')}, height {_chat_number(profile.get('height_cm'), ' cm')}, "
            f"average sleep {sleep_hours}"
            + (f" ({sleep_category}: {sleep_meaning})." if sleep_category else ".")
        )
    elif is_risk_question:
        reply = f"Current AI risk summary: {risk_label}. Recommended slot: {recommended_slot}."
    elif is_appointment_question:
        if upcoming_appointment:
            dt = upcoming_appointment.get("appointment_dt")
            reply = (
                f"Your next appointment is with Dr. {_chat_text(upcoming_appointment.get('doctor_id'))} for "
                f"{_chat_text(upcoming_appointment.get('appointment_type'))} on {_format_date_label(dt)} at {_format_time_label(dt)}."
            )
        elif latest_appointment:
            dt = latest_appointment.get("appointment_dt")
            reply = (
                f"Your latest appointment on record is with Dr. {_chat_text(latest_appointment.get('doctor_id'))} on "
                f"{_format_date_label(dt)} at {_format_time_label(dt)}."
            )
        else:
            reply = "No appointment is currently stored for this patient."
    elif doctor_list_requested:
        if suggested_doctors:
            doctor_lines = [
                f"{idx}. Dr. {item['doctor_name']} ({item['specialization']}) - Available: {item['available_time']}"
                + (" - Emergency doctor" if item.get("emergency_doctor") == "yes" else "")
                for idx, item in enumerate(suggested_doctors, start=1)
            ]
            reply = f"{doctor_reason}\nRecommended doctors for you:\n" + "\n".join(doctor_lines)
        else:
            reply = "I could not find any doctor profiles in the current portal data."
    elif is_precaution_question:
        reply = "Doctor precautions based on your current record: " + " ".join(precautions)
    elif is_general_chat:
        reply = _offline_general_chat_reply(user_message, {
            "profile": {
                "symptoms": symptoms,
                "blood_pressure_systolic": bp_sys,
                "blood_pressure_diastolic": bp_dia,
            },
            "risk_summary": {
                "risk_level": risk_label,
            },
        })
    else:
        reply = (
            f"I can help with your patient details, appointment plan, AI risk summary, and doctor precautions. "
            f"Right now your record shows {risk_label.lower()} with symptoms: {symptoms}. "
            f"Main precautions: {' '.join(precautions[:3])}"
        )

    chat_context = {
        "patient_id": patient_id,
        "patient_name": name,
        "profile": {
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "glucose": glucose,
            "bmi": bmi,
            "blood_pressure_systolic": bp_sys,
            "blood_pressure_diastolic": bp_dia,
            "average_sleep_hours": sleep_hours,
            "sleep_category": sleep_category,
            "sleep_meaning": sleep_meaning,
            "medical_history": _chat_text(profile.get("medical_history")),
            "family_history": _chat_text(profile.get("family_history")),
            "smoking_habit": _chat_text(profile.get("smoking_habit")),
            "alcohol_habit": _chat_text(profile.get("alcohol_habit")),
        },
        "risk_summary": {
            "risk_level": risk_label,
            "recommended_slot": recommended_slot,
        },
        "appointments": {
            "upcoming": {
                "doctor_id": _chat_text(upcoming_appointment.get("doctor_id"), ""),
                "appointment_type": _chat_text(upcoming_appointment.get("appointment_type"), ""),
                "appointment_date": _format_date_label(upcoming_appointment.get("appointment_dt")),
                "appointment_time": _format_time_label(upcoming_appointment.get("appointment_dt")),
            }
            if upcoming_appointment
            else {},
            "latest": {
                "doctor_id": _chat_text(latest_appointment.get("doctor_id"), ""),
                "appointment_type": _chat_text(latest_appointment.get("appointment_type"), ""),
                "appointment_date": _format_date_label(latest_appointment.get("appointment_dt")),
                "appointment_time": _format_time_label(latest_appointment.get("appointment_dt")),
            }
            if latest_appointment
            else {},
        },
        "precautions": precautions,
        "suggested_doctors": suggested_doctors,
        "fallback_reply": reply,
        "question_scope": "general" if is_general_chat else "patient_portal",
    }
    openai_reply = _patient_chat_openai_reply(user_message=user_message, context=chat_context)
    if openai_reply:
        reply = openai_reply
    elif is_general_chat:
        reply = _offline_general_chat_reply(user_message, chat_context)

    should_add_action_suggestion = any(
        (
            _is_health_related_message(user_message),
            is_history_question,
            is_vitals_question,
            is_risk_question,
            is_appointment_question,
            doctor_list_requested,
            is_precaution_question,
        )
    )
    if should_add_action_suggestion:
        reply = _append_action_suggestion(reply, user_message, chat_context)

    return {
        "reply": reply,
        "patient_snapshot": {
            "patient_name": name,
            "age": age,
            "gender": gender,
            "risk_level": risk_label,
            "recommended_slot": recommended_slot,
            "symptoms": symptoms,
            "average_sleep_hours": sleep_hours,
            "sleep_category": sleep_category,
        },
        "precautions": precautions,
        "suggested_doctors": suggested_doctors,
    }



from routes.common import register_common_routes
from routes.patient import register_patient_routes
from routes.staff import register_staff_routes


register_common_routes(app)
register_patient_routes(app)
register_staff_routes(app)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
