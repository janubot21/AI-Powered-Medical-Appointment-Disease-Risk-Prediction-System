from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from doctor_auth import DoctorAuthManager
from paths import (
    DOCTOR_ACCOUNTS_CSV,
    DOCTOR_LEAVE_CSV,
    DOCTOR_PROFILE_CSV,
    NEW_PATIENT_CSV,
    PATIENTS_CSV,
    PATIENT_DB_PATH,
    ensure_csv_exists,
)
from patient_db import PatientDatabase
from predict import LABEL_ENCODER_PATH, MODEL_PATH, RiskEngine, to_jsonable
from triage import determine_priority


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR / "templates"),
    static_folder=str(FRONTEND_DIR / "static"),
    static_url_path="/static",
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")
cookie_secure = os.getenv("COOKIE_SECURE", "0").strip().lower() in {"1", "true", "yes"}
app.config.update(
    TEMPLATES_AUTO_RELOAD=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=cookie_secure,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=8),
)
app.jinja_env.auto_reload = True

doctor_auth_manager = DoctorAuthManager()
risk_engine = RiskEngine(MODEL_PATH, LABEL_ENCODER_PATH)
patient_db = PatientDatabase(PATIENT_DB_PATH)
PASSWORD_POLICY_PATTERN = re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")
PASSWORD_POLICY_MESSAGE = (
    "Password must contain minimum 8 characters, including uppercase (A-Z), lowercase (a-z), number (0-9), and special character (@,!,#,$,%,&,*)."
)
SYMPTOMS_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9,\s./()+:&'-]+")
HEALTH_DATA_SUBMITTED_AT_COLUMN = "Health_Data_Submitted_At"


def _is_patient_authenticated() -> bool:
    return bool(str(session.get("patient_id", "")).strip()) and bool(session.get("patient_authenticated"))


def _is_doctor_authenticated() -> bool:
    return bool(str(session.get("doctor_id", "")).strip()) and bool(session.get("doctor_authenticated"))


def _active_role() -> str:
    if _is_doctor_authenticated():
        return "doctor"
    if _is_patient_authenticated():
        return "patient"
    return "anonymous"


def patient_required(*, api: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            role = _active_role()
            if role == "patient":
                return func(*args, **kwargs)
            if role == "doctor":
                if api:
                    return jsonify({"detail": "Forbidden: doctor account cannot access patient endpoint."}), 403
                return redirect(url_for("doctor_dashboard"))
            if api:
                return jsonify({"detail": "Unauthorized"}), 401
            return redirect(url_for("patient_login"))

        return wrapper

    return decorator


def doctor_required(*, api: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            role = _active_role()
            if role == "doctor":
                return func(*args, **kwargs)
            if role == "patient":
                if api:
                    return jsonify({"detail": "Forbidden: patient account cannot access doctor endpoint."}), 403
                return redirect(url_for("book_appointment"))
            if api:
                return jsonify({"detail": "Unauthorized"}), 401
            return redirect(url_for("doctor_login"))

        return wrapper

    return decorator


def _parse_appointment_time(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


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
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


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


def _load_patients_df() -> pd.DataFrame:
    columns = ["patient_id", "name", "unique_code", "password", "health_details_submitted", "created_at"]
    ensure_csv_exists(PATIENTS_CSV)
    try:
        df = pd.read_csv(PATIENTS_CSV)
    except Exception:
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


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


def _save_patients_df(df: pd.DataFrame) -> None:
    df.to_csv(PATIENTS_CSV, index=False)


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
        "password": password,
        "health_details_submitted": 0,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_patients_df(df)
    return patient_id


def _authenticate_patient(name: str, unique_code: str, password: str) -> Optional[Dict[str, Any]]:
    df = _load_patients_df()
    pwd_mask = df["password"].astype(str) == password
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

    matches = df[identifier_mask & pwd_mask]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


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
    doctor_keywords = ("doctor", "precaution", "advice", "care", "treatment", "medicine")
    appointment_keywords = ("appointment", "booking", "doctor assigned", "doctor name", "visit")
    vitals_keywords = ("bp", "blood pressure", "glucose", "sugar", "bmi", "weight", "height")
    doctor_list_requested = (
        ("doctor" in message or "doctors" in message)
        and any(term in message for term in ("list", "consult", "recommend", "suggest", "available", "avail", "condition", "issue", "show", "who"))
    )

    if any(key in message for key in ("name", "age", "gender", "patient detail", "patient details", "profile")):
        sleep_text = f" Average sleep: {sleep_hours}."
        if sleep_category:
            sleep_text += f" Sleep category: {sleep_category} ({sleep_meaning})."
        reply = f"Patient summary: {name}, age {age}, gender {gender}. Symptoms: {symptoms}.{sleep_text}"
    elif any(key in message for key in ("symptom", "history", "family", "smoking", "alcohol")):
        reply = (
            f"Symptoms: {symptoms}. Medical history: {_chat_text(profile.get('medical_history'))}. "
            f"Family history: {_chat_text(profile.get('family_history'))}. Smoking: {_chat_text(profile.get('smoking_habit'))}. "
            f"Alcohol: {_chat_text(profile.get('alcohol_habit'))}. Average sleep: {sleep_hours}"
            + (f" ({sleep_category}: {sleep_meaning})." if sleep_category else ".")
        )
    elif any(key in message for key in vitals_keywords):
        reply = (
            f"Vitals summary: blood pressure {bp_sys}/{bp_dia} mmHg, glucose {glucose}, BMI {bmi}, "
            f"weight {_chat_number(profile.get('weight_kg'), ' kg')}, height {_chat_number(profile.get('height_cm'), ' cm')}, "
            f"average sleep {sleep_hours}"
            + (f" ({sleep_category}: {sleep_meaning})." if sleep_category else ".")
        )
    elif "risk" in message or "prediction" in message:
        reply = f"Current AI risk summary: {risk_label}. Recommended slot: {recommended_slot}."
    elif any(key in message for key in appointment_keywords):
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
    elif any(key in message for key in doctor_keywords):
        reply = "Doctor precautions based on your current record: " + " ".join(precautions)
    else:
        reply = (
            f"I can help with your patient details, appointment plan, AI risk summary, and doctor precautions. "
            f"Right now your record shows {risk_label.lower()} with symptoms: {symptoms}. "
            f"Main precautions: {' '.join(precautions[:3])}"
        )

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


@app.route("/")
def home() -> Any:
    return redirect(url_for("role_login"))


@app.route("/login")
def role_login() -> Any:
    return render_template("flask_role_login.html")


@app.route("/about")
def about_page() -> Any:
    return render_template("flask_about.html")


@app.route("/patient/signup", methods=["GET", "POST"])
def patient_signup() -> Any:
    errors: list[str] = []
    form_data = {"name": ""}

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        form_data["name"] = name

        if not name:
            errors.append("Patient Name is required.")
        if not PASSWORD_POLICY_PATTERN.fullmatch(password):
            errors.append(PASSWORD_POLICY_MESSAGE)
        if password != confirm_password:
            errors.append("Password and Confirm Password must match.")

        if not errors:
            try:
                patient_id = _create_patient_account(name, "", password)
                session.pop("doctor_id", None)
                session.pop("doctor_name", None)
                session.pop("doctor_authenticated", None)
                session["patient_id"] = patient_id
                session["patient_name"] = name
                session["patient_authenticated"] = False
                session["role"] = "patient"
                session["allow_health_details"] = True
                return redirect(url_for("health_details", patient_id=patient_id))
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Signup failed: {exc}")

    return render_template("flask_patient_signup.html", errors=errors, form_data=form_data)


@app.route("/patient/health-details", methods=["GET", "POST"])
def health_details() -> Any:
    errors: list[str] = []
    if not session.get("allow_health_details"):
        patient_id = session.get("patient_id")
        if patient_id:
            return redirect(url_for("book_appointment"))
        return redirect(url_for("patient_signup"))

    patient_id = (
        session.get("patient_id", "")
        or request.args.get("patient_id", "").strip()
        or request.form.get("patient_id", "").strip()
    )
    form_data = {
        "patient_id": patient_id,
        "age": "",
        "gender": "",
        "symptoms": "",
        "symptom_count": "",
        "glucose": "",
        "blood_pressure": "",
        "height_cm": "",
        "weight_kg": "",
        "bmi": "",
        "smoking_habit": "",
        "alcohol_habit": "",
        "average_sleep_hours": "",
        "medical_history": "",
        "family_history": "",
    }

    if request.method == "POST":
        form_data.update(
            {
                "age": request.form.get("age", "").strip(),
                "gender": request.form.get("gender", "").strip().lower(),
                "symptoms": _sanitize_symptoms_text(request.form.get("symptoms", "")),
                "symptom_count": request.form.get("symptom_count", "").strip(),
                "glucose": request.form.get("glucose", "").strip(),
                "blood_pressure": request.form.get("blood_pressure", "").strip(),
                "height_cm": request.form.get("height_cm", "").strip(),
                "weight_kg": request.form.get("weight_kg", "").strip(),
                "bmi": request.form.get("bmi", "").strip(),
                "smoking_habit": request.form.get("smoking_habit", "").strip().lower(),
                "alcohol_habit": request.form.get("alcohol_habit", "").strip().lower(),
                "average_sleep_hours": request.form.get("average_sleep_hours", "").strip(),
                "medical_history": request.form.get("medical_history", "").strip(),
                "family_history": request.form.get("family_history", "").strip().lower(),
            }
        )

        if not patient_id:
            errors.append("Patient ID is missing. Please signup again.")
        if form_data["gender"] not in {"male", "female", "other"}:
            errors.append("Gender must be one of: male, female, other.")
        if form_data["smoking_habit"] not in {"yes", "no"}:
            errors.append("Smoking Habit must be yes or no.")
        if form_data["alcohol_habit"] not in {"yes", "no"}:
            errors.append("Alcohol Habit must be yes or no.")
        if form_data["family_history"] not in {"yes", "no"}:
            errors.append("Family History must be yes or no.")
        if not form_data["symptoms"]:
            errors.append("Symptoms are required.")

        try:
            age = _to_int(form_data["age"], "Age", 0, 130)
            symptom_count = _to_int(form_data["symptom_count"], "Symptom_Count", 0, 100)
            glucose = _to_float(form_data["glucose"], "Glucose", 0, 1000)
            blood_pressure = _to_float(form_data["blood_pressure"], "BloodPressure", 0, 400)
            height_cm = _to_float(form_data["height_cm"], "Height (cm)", 1, 300)
            weight_kg = _to_float(form_data["weight_kg"], "Weight (kg)", 1, 500)
            average_sleep_hours = _normalize_average_sleep_hours(form_data["average_sleep_hours"], required=True)
            bmi = _calculate_bmi(height_cm, weight_kg)
            form_data["bmi"] = f"{bmi:.2f}"
        except ValueError as exc:
            errors.append(str(exc))
            age = symptom_count = 0
            glucose = blood_pressure = bmi = average_sleep_hours = 0.0
            height_cm = weight_kg = 0.0

        if not errors:
            row = {
                "Patient_ID": patient_id,
                "Age": age,
                "Gender": form_data["gender"],
                "Symptoms": form_data["symptoms"],
                "Symptom_Count": symptom_count,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "BMI": bmi,
                "Height_cm": height_cm,
                "Weight_kg": weight_kg,
                "Smoking_Habit": form_data["smoking_habit"],
                "Alcohol_Habit": form_data["alcohol_habit"],
                "Average_Sleep_Hours": average_sleep_hours,
                "Medical_History": form_data["medical_history"],
                "Family_History": form_data["family_history"],
                HEALTH_DATA_SUBMITTED_AT_COLUMN: _utc_now_iso(),
            }
            try:
                append_to_new_patient_csv(row)
                _mark_health_details_submitted(patient_id)
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Could not save health details: {exc}")

            if not errors:
                session["health_confirmation"] = row
                session.pop("patient_id", None)
                session.pop("patient_name", None)
                session.pop("allow_health_details", None)
                return redirect(url_for("health_confirmation"))

    return render_template("flask_health_details.html", errors=errors, form_data=form_data)


@app.route("/patient/health-confirmation")
def health_confirmation() -> Any:
    row = session.get("health_confirmation")
    if not row:
        return redirect(url_for("patient_signup"))
    return render_template("flask_health_confirmation.html", row=row)


@app.route("/patient/login", methods=["GET", "POST"])
def patient_login() -> Any:
    errors: list[str] = []
    form_data = {"patient_name": ""}

    if request.method == "GET":
        # Force explicit patient re-auth when user opens patient login route.
        session.pop("patient_id", None)
        session.pop("patient_name", None)
        session.pop("patient_authenticated", None)
        if session.get("role") == "patient":
            session.pop("role", None)

    if request.method == "POST":
        patient_name = request.form.get("patient_name", "").strip()
        password = request.form.get("password", "").strip()
        form_data["patient_name"] = patient_name
        if not patient_name:
            errors.append("Enter Patient Name.")
        if not password:
            errors.append("Password is required.")

        if not errors:
            try:
                patient = _authenticate_patient(patient_name, "", password)
                if not patient:
                    raise ValueError("Invalid patient name or password.")
                patient_id = str(patient.get("patient_id", "")).strip()
                # Patient login should not inherit doctor portal access.
                session.pop("doctor_id", None)
                session.pop("doctor_name", None)
                session.pop("doctor_authenticated", None)
                session["patient_id"] = patient_id
                session["patient_name"] = patient_name
                session["patient_authenticated"] = True
                session["role"] = "patient"
                session.pop("health_confirmation", None)
                session.pop("allow_health_details", None)
                return render_template(
                    "flask_patient_login.html",
                    errors=[],
                    form_data={"patient_name": ""},
                    login_success=True,
                    redirect_url=url_for("book_appointment"),
                )
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Login failed: {exc}")

    return render_template("flask_patient_login.html", errors=errors, form_data=form_data)


@app.route("/patient/book-appointment")
@patient_required()
def book_appointment() -> Any:
    patient_id = session.get("patient_id")
    patient_name = str(session.get("patient_name", "")).strip()
    if not patient_name:
        patient_name = _get_patient_name_by_id(str(patient_id).strip())
    doctor_ids = _load_doctor_ids()
    return render_template(
        "flask_book_appointment.html",
        patient_id=patient_id,
        doctor_ids=doctor_ids,
        patient_name=patient_name,
    )


@app.route("/patient/ai-chat")
@patient_required()
def patient_ai_chat() -> Any:
    patient_id = str(session.get("patient_id", "")).strip()
    patient_name = str(session.get("patient_name", "")).strip() or _get_patient_name_by_id(patient_id) or "Patient"
    initial_payload = _patient_chat_response(patient_id, "Give my patient summary and precautions.")
    return render_template(
        "flask_patient_ai_chat.html",
        patient_id=patient_id,
        patient_name=patient_name,
        initial_reply=initial_payload.get("reply", ""),
        initial_snapshot=initial_payload.get("patient_snapshot", {}),
        initial_precautions=initial_payload.get("precautions", []),
        initial_suggested_doctors=initial_payload.get("suggested_doctors", []),
    )


@app.post("/patient/ai-chat/message")
@patient_required(api=True)
def patient_ai_chat_message() -> Any:
    patient_id = str(session.get("patient_id", "")).strip()
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return jsonify({"detail": "message is required"}), 400
    return jsonify(_patient_chat_response(patient_id, message))


@app.route("/patient/logout")
def patient_logout() -> Any:
    session.pop("patient_id", None)
    session.pop("patient_name", None)
    session.pop("patient_authenticated", None)
    if session.get("role") == "patient":
        session.pop("role", None)
    session.pop("health_confirmation", None)
    session.pop("allow_health_details", None)
    return redirect(url_for("role_login"))


@app.route("/patient/booking-confirmation")
@patient_required()
def patient_booking_confirmation() -> Any:
    patient_id = str(session.get("patient_id", "")).strip()

    patient_name = str(session.get("patient_name", "")).strip() or _get_patient_name_by_id(patient_id)
    normalized_pid = _normalize_patient_id(patient_id)

    patient_appointments: list[Dict[str, Any]] = []
    stored_appointments = patient_db.list_appointments(patient_id=patient_id)
    for item in stored_appointments:
        if _normalize_patient_id(item.get("patient_id")) != normalized_pid:
            continue
        dt = _parse_appointment_time(item.get("appointment_time"))
        patient_appointments.append(
            {
                "appointment_id": str(item.get("appointment_id", "")),
                "doctor_id": str(item.get("doctor_id", "")).strip(),
                "appointment_type": str(item.get("appointment_type", "")).strip(),
                "appointment_time": dt,
                "date_label": _format_date_label(dt),
                "time_label": _format_time_label(dt),
                "appointment_priority": str(item.get("appointment_priority", "")).strip() or "Normal",
                "recommended_slot": str(item.get("recommended_slot", "")).strip() or "Next Available Date",
                "priority_badge_text": str(item.get("priority_badge_text", "")).strip() or "Normal Appointment",
                "priority_badge_color": str(item.get("priority_badge_color", "")).strip() or "green",
            }
        )

    patient_appointments.sort(key=lambda x: x.get("appointment_time") or datetime.min, reverse=True)

    now = datetime.now()
    upcoming = [a for a in patient_appointments if a.get("appointment_time") and a["appointment_time"] >= now]
    past = [a for a in patient_appointments if not a.get("appointment_time") or a["appointment_time"] < now]
    upcoming.sort(key=lambda x: x.get("appointment_time") or datetime.max)

    patient_features: Dict[str, Any] = {}
    if patient_appointments:
        latest_source = next(
            (
                ap
                for ap in stored_appointments
                if _normalize_patient_id(ap.get("patient_id")) == normalized_pid and isinstance(ap.get("patient_features"), dict)
            ),
            None,
        )
        if latest_source:
            patient_features = dict(latest_source.get("patient_features") or {})
    if not patient_features:
        try:
            patient_features = get_features_for_patient(patient_id)
        except ValueError:
            patient_features = {}

    return render_template(
        "flask_booking_confirmation.html",
        patient_id=patient_id,
        patient_name=patient_name,
        patient_features=patient_features,
        upcoming_appointments=upcoming,
        past_appointments=past,
    )


@app.post("/patient/appointments/cancel")
@patient_required(api=True)
def patient_cancel_appointment() -> Any:
    patient_id = str(session.get("patient_id", "")).strip()

    appointment_id = str(request.form.get("appointment_id", "")).strip()
    if not appointment_id:
        return redirect(url_for("patient_booking_confirmation"))

    patient_db.delete_appointment(appointment_id, patient_id=patient_id)

    return redirect(url_for("patient_booking_confirmation"))


@app.get("/patient/features/<patient_id>")
def patient_features(patient_id: str) -> Any:
    if _is_patient_authenticated():
        session_pid = _normalize_patient_id(session.get("patient_id"))
        requested_pid = _normalize_patient_id(patient_id)
        if session_pid != requested_pid:
            return jsonify({"detail": "Forbidden: you can only access your own patient record."}), 403
    elif not _is_doctor_authenticated():
        return jsonify({"detail": "Unauthorized"}), 401

    try:
        features = get_features_for_patient(patient_id)
        return jsonify({"patient_id": patient_id, "patient_features": features})
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 404
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Patient lookup failed: {exc}"}), 500


@app.post("/patient/features/<patient_id>")
@patient_required(api=True)
def update_patient_features(patient_id: str) -> Any:
    session_pid = _normalize_patient_id(session.get("patient_id"))
    requested_pid = _normalize_patient_id(patient_id)
    if not session_pid or requested_pid != session_pid:
        return jsonify({"detail": "Forbidden: you can only update your own patient record."}), 403

    payload = request.get_json(silent=True) or {}
    patient_features = payload.get("patient_features")
    if not isinstance(patient_features, dict) or not patient_features:
        return jsonify({"detail": "patient_features is required"}), 400

    try:
        upsert_new_patient_csv_from_features(patient_id, patient_features)
        refreshed_features = get_features_for_patient(patient_id)
        return jsonify({"patient_id": patient_id, "patient_features": refreshed_features})
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Could not update patient features: {exc}"}), 500


@app.post("/patient/book-appointment-submit")
@patient_required(api=True)
def submit_appointment() -> Any:
    payload = request.get_json(silent=True) or {}
    patient_id = str(payload.get("patient_id", "")).strip()
    patient_name = str(payload.get("patient_name", "")).strip()
    contact_info = str(payload.get("contact_info", "")).strip()
    doctor_id = str(payload.get("doctor_id", "")).strip()
    appointment_type = str(payload.get("appointment_type", "")).strip()
    appointment_time = str(payload.get("appointment_time", "")).strip()
    patient_features = payload.get("patient_features")

    if not patient_id:
        return jsonify({"detail": "patient_id is required"}), 400
    if not patient_name:
        return jsonify({"detail": "patient_name is required"}), 400
    if not contact_info:
        return jsonify({"detail": "contact_info is required"}), 400
    if not doctor_id:
        return jsonify({"detail": "doctor_id is required"}), 400
    if not appointment_type:
        return jsonify({"detail": "appointment_type is required"}), 400
    if not appointment_time:
        return jsonify({"detail": "appointment_time is required"}), 400

    session_pid = _normalize_patient_id(session.get("patient_id"))
    payload_pid = _normalize_patient_id(patient_id)
    if not session_pid or payload_pid != session_pid:
        return jsonify({"detail": "Forbidden: patient_id does not match your logged-in session."}), 403

    try:
        parsed_time = datetime.fromisoformat(appointment_time)
    except ValueError:
        return jsonify({"detail": "appointment_time must be ISO format"}), 400

    if patient_features is None:
        try:
            patient_features = get_features_for_patient(patient_id)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400

    try:
        upsert_new_patient_csv_from_features(patient_id, patient_features)
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Could not update patient CSV: {exc}"}), 500

    try:
        risk_assessment = risk_engine.predict(patient_features)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Risk assessment failed: {exc}"}), 500

    risk_level_text = str(risk_assessment.risk_level or "").strip().lower()
    selected_doctor_profile = _doctor_profile_by_id(doctor_id)

    if _is_night_emergency_window(parsed_time) and not _is_emergency_doctor_profile(selected_doctor_profile):
        emergency_doctors = _available_emergency_doctors(parsed_time, exclude_doctor_id=doctor_id)
        alternative = emergency_doctors[0] if emergency_doctors else None
        return jsonify(
            {
                "booking_status": "doctor_unavailable",
                "reason": "non_emergency_doctor_night_window",
                "risk_level": risk_level_text.title() or "Low",
                "message": "Only emergency doctors are available during night hours (7:00 PM to 7:00 AM). Please choose an emergency doctor.",
                "selected_doctor": selected_doctor_profile,
                "alternative_doctor": alternative,
                "emergency_doctors_available": emergency_doctors,
                "can_book_alternative": bool(alternative),
            }
        )

    if _is_doctor_on_leave(doctor_id, parsed_time):
        if risk_level_text == "high":
            emergency_doctors = _available_emergency_doctors(parsed_time, exclude_doctor_id=doctor_id)
            alternative = emergency_doctors[0] if emergency_doctors else None
            if alternative:
                return jsonify(
                    {
                        "booking_status": "doctor_unavailable",
                        "reason": "doctor_on_leave",
                        "risk_level": "High",
                        "message": "Selected doctor is on leave. Because your condition is high risk, we recommend consulting an available emergency doctor.",
                        "selected_doctor": selected_doctor_profile,
                        "alternative_doctor": alternative,
                        "emergency_doctors_available": emergency_doctors,
                        "can_book_alternative": True,
                    }
                )
            return jsonify(
                {
                    "booking_status": "doctor_unavailable",
                    "reason": "doctor_on_leave",
                    "risk_level": "High",
                    "message": "Selected doctor is on leave. Because your condition is high risk, no emergency doctor is currently available on this date.",
                    "selected_doctor": selected_doctor_profile,
                    "alternative_doctor": None,
                    "emergency_doctors_available": [],
                    "can_book_alternative": False,
                }
            )

        return jsonify(
            {
                "booking_status": "doctor_unavailable",
                "reason": "doctor_on_leave",
                "risk_level": risk_level_text.title() or "Low",
                "message": "Selected doctor is on leave. Please choose another date or another doctor.",
                "selected_doctor": selected_doctor_profile,
                "alternative_doctor": None,
                "emergency_doctors_available": [],
                "can_book_alternative": False,
            }
        )

    priority = determine_priority(risk_assessment.risk_level)
    appointment_id = uuid4().hex
    result = {
        "booking_status": "confirmed",
        "appointment_id": appointment_id,
        "patient_id": patient_id,
        "patient_name": patient_name,
        "contact_info": contact_info,
        "doctor_id": doctor_id,
        "appointment_type": appointment_type,
        "appointment_time": parsed_time.isoformat(),
        "predicted_disease": risk_assessment.predicted_class,
        "risk_level": risk_assessment.risk_level.title(),
        "appointment_priority": priority.priority,
        "recommended_slot": priority.recommended_slot,
        "priority_badge_text": priority.badge_text,
        "priority_badge_color": priority.badge_color,
        "redirect_url": url_for("patient_booking_confirmation"),
    }
    patient_db.add_appointment(
        {
            "appointment_id": appointment_id,
            "booked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "patient_id": patient_id,
            "patient_name": patient_name,
            "contact_info": contact_info,
            "doctor_id": doctor_id,
            "appointment_type": appointment_type,
            "appointment_time": parsed_time.isoformat(),
            "patient_features": to_jsonable(patient_features),
            "risk_assessment": risk_assessment.model_dump(),
            "appointment_priority": priority.priority,
            "recommended_slot": priority.recommended_slot,
            "priority_badge_text": priority.badge_text,
            "priority_badge_color": priority.badge_color,
        }
    )
    return jsonify(result)


@app.route("/doctor/signup", methods=["GET", "POST"])
def doctor_signup() -> Any:
    errors: list[str] = []
    form_data = {"doctor_id": ""}

    if request.method == "POST":
        doctor_id = request.form.get("doctor_id", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        form_data["doctor_id"] = doctor_id

        if not doctor_id:
            errors.append("Doctor ID is required.")
        if not PASSWORD_POLICY_PATTERN.fullmatch(password):
            errors.append(PASSWORD_POLICY_MESSAGE)
        if password != confirm_password:
            errors.append("Password and Confirm Password must match.")

        if not errors:
            try:
                doctor_auth_manager.signup(doctor_id, password)
                return redirect(url_for("doctor_login"))
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Signup failed: {exc}")

    return render_template("flask_doctor_signup.html", errors=errors, form_data=form_data)


@app.route("/doctor/login", methods=["GET", "POST"])
def doctor_login() -> Any:
    errors: list[str] = []
    form_data = {"doctor_id": ""}

    if request.method == "GET":
        # Force explicit doctor re-auth when user opens doctor login route.
        session.pop("doctor_id", None)
        session.pop("doctor_name", None)
        session.pop("doctor_authenticated", None)
        if session.get("role") == "doctor":
            session.pop("role", None)

    if request.method == "POST":
        doctor_id = request.form.get("doctor_id", "").strip()
        password = request.form.get("password", "").strip()
        form_data["doctor_id"] = doctor_id
        if not doctor_id:
            errors.append("Enter Doctor ID.")
        if not password:
            errors.append("Password is required.")

        if not errors:
            try:
                resolved_doctor_id = doctor_auth_manager.login(doctor_id, "", "", password)
                # Doctor login should not inherit patient portal access.
                session.pop("patient_id", None)
                session.pop("patient_name", None)
                session.pop("patient_authenticated", None)
                session["doctor_id"] = resolved_doctor_id
                session["doctor_name"] = resolved_doctor_id
                session["doctor_authenticated"] = True
                session["role"] = "doctor"
                return render_template(
                    "flask_doctor_login.html",
                    errors=[],
                    form_data={"doctor_id": ""},
                    login_success=True,
                    redirect_url=url_for("doctor_dashboard"),
                )
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Login failed: {exc}")

    return render_template("flask_doctor_login.html", errors=errors, form_data=form_data)


@app.route("/doctor/dashboard")
@doctor_required()
def doctor_dashboard() -> Any:
    doctor_id = session.get("doctor_id")
    doctor_name = str(session.get("doctor_name", "")).strip() or str(doctor_id)
    return render_template("flask_doctor_dashboard.html", doctor_id=doctor_id, doctor_name=doctor_name)


@app.route("/doctor/patient-database-page")
@doctor_required()
def doctor_patient_database_page() -> Any:
    doctor_id = session.get("doctor_id")
    doctor_name = str(session.get("doctor_name", "")).strip() or str(doctor_id)
    return render_template(
        "flask_doctor_patient_database.html",
        doctor_id=doctor_id,
        doctor_name=doctor_name,
    )


@app.route("/doctor/logout")
def doctor_logout() -> Any:
    session.pop("doctor_id", None)
    session.pop("doctor_name", None)
    session.pop("doctor_authenticated", None)
    if session.get("role") == "doctor":
        session.pop("role", None)
    return redirect(url_for("role_login"))


@app.get("/doctor/appointments")
@doctor_required(api=True)
def doctor_appointments() -> Any:
    current_doctor_id = _normalize_doctor_id(session.get("doctor_id"))
    appointments = patient_db.list_appointments(doctor_id=current_doctor_id)
    appointments = sorted(appointments, key=_doctor_appointment_sort_key)
    return jsonify({"appointments": appointments})


@app.get("/doctor/patient-database")
@doctor_required(api=True)
def doctor_patient_database() -> Any:
    current_doctor_id = _normalize_doctor_id(session.get("doctor_id"))
    doctor_patient_ids = {
        _normalize_patient_id(ap.get("patient_id"))
        for ap in patient_db.list_appointments(doctor_id=current_doctor_id)
        if _normalize_doctor_id(ap.get("doctor_id")) == current_doctor_id
    }
    if not doctor_patient_ids:
        return jsonify({"patients": []})

    patients = []
    for row in patient_db.list_profiles():
        if _normalize_patient_id(row.get("patient_id")) in doctor_patient_ids:
            patients.append(row)
    return jsonify({"patients": patients})


@app.get("/doctor/leaves")
@doctor_required(api=True)
def doctor_leaves() -> Any:
    doctor_id = str(request.args.get("doctor_id", "")).strip()
    rows = _load_doctor_leave_df().to_dict(orient="records")
    if doctor_id:
        doctor_norm = _normalize_doctor_id(doctor_id)
        rows = [r for r in rows if _normalize_doctor_id(r.get("doctor_id", "")) == doctor_norm]
    return jsonify({"leaves": rows})


@app.post("/doctor/leaves")
@doctor_required(api=True)
def upsert_doctor_leave() -> Any:
    payload = request.get_json(silent=True) or {}
    leave_date = str(payload.get("leave_date", "")).strip()
    doctor_id = str(payload.get("doctor_id", "")).strip() or str(session.get("doctor_id", "")).strip()
    if not doctor_id:
        return jsonify({"detail": "doctor_id is required"}), 400
    if not leave_date:
        return jsonify({"detail": "leave_date is required"}), 400
    try:
        _add_doctor_leave(doctor_id, leave_date)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    return jsonify({"status": "ok", "doctor_id": doctor_id, "leave_date": leave_date[:10]})


@app.get("/doctor/availability-status")
@doctor_required(api=True)
def doctor_availability_status() -> Any:
    doctor_id = str(request.args.get("doctor_id", "")).strip() or str(session.get("doctor_id", "")).strip()
    if not doctor_id:
        return jsonify({"detail": "doctor_id is required"}), 400
    date_text = str(request.args.get("date", "")).strip()
    date_key = date_text[:10] if date_text else datetime.now().date().isoformat()
    try:
        parsed = datetime.fromisoformat(date_key)
    except ValueError:
        return jsonify({"detail": "date must be YYYY-MM-DD"}), 400
    day = parsed.date().isoformat()
    status = "leave" if _is_doctor_on_leave(doctor_id, parsed) else "available"
    return jsonify({"doctor_id": doctor_id, "date": day, "status": status})


@app.post("/doctor/availability-status")
@doctor_required(api=True)
def set_doctor_availability_status() -> Any:
    payload = request.get_json(silent=True) or {}
    doctor_id = str(payload.get("doctor_id", "")).strip() or str(session.get("doctor_id", "")).strip()
    if not doctor_id:
        return jsonify({"detail": "doctor_id is required"}), 400
    status = str(payload.get("status", "")).strip().lower()
    if status not in {"available", "leave"}:
        return jsonify({"detail": "status must be 'available' or 'leave'"}), 400
    date_text = str(payload.get("date", "")).strip()
    date_key = date_text[:10] if date_text else datetime.now().date().isoformat()
    try:
        parsed = datetime.fromisoformat(date_key)
    except ValueError:
        return jsonify({"detail": "date must be YYYY-MM-DD"}), 400
    day = parsed.date().isoformat()
    try:
        if status == "leave":
            _add_doctor_leave(doctor_id, day)
        else:
            _remove_doctor_leave(doctor_id, day)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    return jsonify({"doctor_id": doctor_id, "date": day, "status": status})


@app.post("/patient/predict-risk")
@patient_required(api=True)
def patient_predict_risk() -> Any:
    payload = request.get_json(silent=True) or {}
    patient_features = payload.get("patient_features")
    if not patient_features:
        return jsonify({"detail": "patient_features is required"}), 400

    try:
        result = risk_engine.predict(patient_features)
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
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Prediction failed: {exc}"}), 500


@app.get("/patient/emergency-doctors")
@patient_required(api=True)
def patient_emergency_doctors() -> Any:
    date_text = str(request.args.get("date", "")).strip()
    time_text = str(request.args.get("time", "")).strip()
    if not date_text:
        return jsonify({"detail": "date is required (YYYY-MM-DD)"}), 400
    if not time_text:
        return jsonify({"detail": "time is required (HH:MM)"}), 400
    try:
        parsed = datetime.fromisoformat(f"{date_text}T{time_text}")
    except ValueError:
        return jsonify({"detail": "Invalid date/time format"}), 400

    is_night_window = _is_night_emergency_window(parsed)
    doctors = _available_emergency_doctors(parsed) if is_night_window else []
    return jsonify(
        {
            "date": parsed.date().isoformat(),
            "time": f"{parsed.hour:02d}:{parsed.minute:02d}",
            "window_active": is_night_window,
            "emergency_doctors": doctors,
        }
    )


@app.post("/doctor/predict-risk")
@doctor_required(api=True)
def doctor_predict_risk() -> Any:
    payload = request.get_json(silent=True) or {}
    patient_features = payload.get("patient_features")
    if not patient_features:
        return jsonify({"detail": "patient_features is required"}), 400

    try:
        result = risk_engine.predict(patient_features)
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
        return jsonify(payload)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
