from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from backend.services.shared import (
    DATA_LOCK,
    SETTINGS,
    HealthDetailsRequest,
    LoginRequest,
    PasswordResetConfirm,
    PasswordResetRequest,
    PatientRecordUpdateRequest,
    RegisterRequest,
    append_health_details_submission,
    append_patient_record_history,
    compute_health_details_changes,
    create_user,
    dashboard_path_for_role,
    dbmod,
    get_db,
    get_user_by_email,
    get_user_by_phone,
    hash_password,
    normalize_email,
    normalize_phone,
    normalize_role,
    rate_limit,
    require_role,
    require_session,
    safe_normalize_role,
    snapshot_health_details,
    split_symptoms_text,
    update_user_health_details,
    user_public_payload,
    validate_full_name,
    validate_email,
    validate_password_strength,
    verify_password,
)

router = APIRouter()


@router.post("/auth/register")
def register_user(payload: RegisterRequest, request: Request) -> dict[str, Any]:
    rate_limit(request, "auth_register")
    full_name = validate_full_name(payload.full_name)
    email = validate_email(payload.email)
    phone = normalize_phone(payload.phone)
    role = normalize_role(payload.role)
    validate_password_strength(payload.password)

    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    if get_user_by_phone(phone):
        raise HTTPException(status_code=409, detail="An account with this phone number already exists.")

    salt = secrets.token_hex(16)
    create_user(
        full_name=full_name,
        email=email,
        phone=phone,
        role=role,
        salt=salt,
        password_hash=hash_password(payload.password, salt),
    )

    # Start session after successful registration.
    user = get_user_by_email(email)
    if user:
        request.session["user"] = user_public_payload(user)
        request.session["csrf"] = secrets.token_hex(16)

    return {
        "message": "Registration successful.",
        "redirect_to": dashboard_path_for_role(role, has_health_details=False),
        "csrf_token": str(request.session.get("csrf") or ""),
        "user": {
            "full_name": full_name,
            "email": email,
            "phone": phone,
            "role": role,
            "dashboard_path": dashboard_path_for_role(role, has_health_details=False),
            "health_details_completed": False,
        },
    }


@router.post("/auth/login")
def login_user(payload: LoginRequest, request: Request) -> dict[str, Any]:
    rate_limit(request, "auth_login")
    role = normalize_role(payload.role)

    user: dict[str, Any] | None = None
    if payload.phone and payload.phone.strip():
        phone = normalize_phone(payload.phone)
        user = get_user_by_phone(phone)
    elif payload.email and payload.email.strip():
        email = validate_email(payload.email)
        user = get_user_by_email(email)
    else:
        raise HTTPException(status_code=400, detail="Provide email or phone number to login.")

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    if user.get("role", "patient") != role:
        raise HTTPException(status_code=403, detail="This account is registered for a different role.")

    if not verify_password(payload.password, user.get("salt") or "", user.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    request.session["user"] = user_public_payload(user)
    request.session["csrf"] = secrets.token_hex(16)

    return {
        "message": "Login successful.",
        "redirect_to": dashboard_path_for_role(role, has_health_details=bool(user.get("health_details"))),
        "csrf_token": str(request.session.get("csrf") or ""),
        "user": user_public_payload(user),
    }


@router.post("/auth/logout")
def logout_user(request: Request) -> dict[str, Any]:
    require_session(request)
    try:
        request.session.clear()
    except Exception:
        pass
    return {"message": "Logged out."}


@router.post("/auth/password-reset/request")
def password_reset_request(payload: PasswordResetRequest, request: Request) -> dict[str, Any]:
    rate_limit(request, "pw_reset_request")
    email = validate_email(payload.email)
    user = get_user_by_email(email)
    # Do not reveal whether the email exists.
    token = secrets.token_urlsafe(24)
    expires_at = (datetime.now(timezone.utc).timestamp() + 30 * 60)
    expires_iso = datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()
    if user:
        try:
            dbmod.create_password_reset(get_db(), email, token, expires_iso)
        except Exception:
            pass

    resp: dict[str, Any] = {"message": "If the account exists, a reset link/token will be issued."}
    if SETTINGS.env not in {"production", "prod"}:
        resp["debug_token"] = token
        resp["expires_at"] = expires_iso
    return resp


@router.post("/auth/password-reset/confirm")
def password_reset_confirm(payload: PasswordResetConfirm, request: Request) -> dict[str, Any]:
    rate_limit(request, "pw_reset_confirm")
    email = validate_email(payload.email)
    validate_password_strength(payload.new_password)

    ok = dbmod.consume_password_reset(get_db(), email, payload.token)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token.")

    salt = secrets.token_hex(16)
    password_hash = hash_password(payload.new_password, salt)
    dbmod.update_user_password(get_db(), email, salt, password_hash)
    return {"message": "Password updated successfully."}


@router.get("/auth/user")
def get_user_public(request: Request, email: str = Query(..., min_length=5, max_length=254)) -> dict[str, Any]:
    session_user = require_session(request)
    target = validate_email(email)

    if str(session_user.get("role") or "") == "patient":
        if normalize_email(str(session_user.get("email") or "")) != target:
            raise HTTPException(status_code=403, detail="Patients can only access their own account.")

    user = get_user_by_email(target)
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")
    return {"user": user_public_payload(user)}


@router.get("/auth/me")
def auth_me(request: Request) -> dict[str, Any]:
    user = require_session(request)
    return {"user": user}


@router.get("/auth/health-details")
def get_health_details(request: Request, email: str = Query(..., min_length=5, max_length=254)) -> dict[str, Any]:
    session_user = require_session(request)
    target = validate_email(email)

    if str(session_user.get("role") or "") == "patient":
        if normalize_email(str(session_user.get("email") or "")) != target:
            raise HTTPException(status_code=403, detail="Patients can only access their own health details.")

    user = get_user_by_email(target)

    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    return {
        "user": user_public_payload(user),
        "health_details": user.get("health_details"),
    }


@router.post("/auth/health-details")
def save_health_details(request: Request, payload: HealthDetailsRequest) -> dict[str, Any]:
    session_user = require_role("patient")(require_session(request))
    if normalize_email(str(session_user.get("email") or "")) != validate_email(payload.email):
        raise HTTPException(status_code=403, detail="You can only update your own health details.")

    user = get_user_by_email(validate_email(payload.email))

    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    before_details = user.get("health_details") if isinstance(user.get("health_details"), dict) else None
    health_details = payload.model_dump()
    health_details["email"] = normalize_email(payload.email)
    changed_at = datetime.now(timezone.utc).isoformat()
    health_details["submitted_at"] = changed_at

    # Patients should not be able to set clinical-measured fields via the UI.
    # Preserve any existing values; allow them to be null if not yet recorded.
    if user.get("role") == "patient":
        existing = user.get("health_details") or {}
        health_details["glucose"] = existing.get("glucose")
        health_details["blood_pressure"] = existing.get("blood_pressure")

    with DATA_LOCK:
        append_health_details_submission(health_details)

    update_user_health_details(payload.email, health_details)
    user["health_details"] = health_details

    # Audit trail: track what changed (patient self-submission / updates).
    try:
        changes = compute_health_details_changes(before_details, health_details)
        if changes:
            append_patient_record_history(
                {
                    "patient_email": normalize_email(payload.email),
                    "changed_at": changed_at,
                    "changed_by_email": normalize_email(payload.email),
                    "changed_by_role": safe_normalize_role(user.get("role") or "patient"),
                    "source": "patient_form",
                    "changes": changes,
                    "before": snapshot_health_details(before_details),
                    "after": snapshot_health_details(health_details),
                }
            )
    except Exception:
        pass

    return {
        "message": "Health details saved successfully.",
        "redirect_to": dashboard_path_for_role(user.get("role", "patient"), has_health_details=True),
        "user": user_public_payload(user),
        "health_details": user["health_details"],
    }


@router.get("/api/patient/history")
def patient_record_history(
    request: Request,
    email: str = Query(..., min_length=5, max_length=254),
    limit: int = Query(20, ge=1, le=200),
) -> dict[str, Any]:
    session_user = require_session(request)
    normalized = validate_email(email)

    if str(session_user.get("role") or "") == "patient":
        if normalize_email(str(session_user.get("email") or "")) != normalized:
            raise HTTPException(status_code=403, detail="Patients can only access their own history.")
    elif str(session_user.get("role") or "") not in {"nurse", "doctor"}:
        raise HTTPException(status_code=403, detail="Forbidden.")

    events = dbmod.list_history_events(get_db(), normalized, limit=int(limit))
    return {"patient_email": normalized, "events": events}


@router.post("/api/patient/health-details/update")
def update_patient_health_details(request: Request, payload: PatientRecordUpdateRequest) -> dict[str, Any]:
    session_user = require_role("nurse", "doctor")(require_session(request))

    patient_email = validate_email(payload.patient_email)
    updated_by_email = normalize_email(str(session_user.get("email") or ""))
    updated_by_role = safe_normalize_role(str(session_user.get("role") or "nurse"))

    changed_at = datetime.now(timezone.utc).isoformat()

    patient = get_user_by_email(patient_email)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient account not found.")

    before = patient.get("health_details") if isinstance(patient.get("health_details"), dict) else {}
    after = dict(before) if isinstance(before, dict) else {}

    if payload.age is not None:
        after["age"] = int(payload.age)
    if payload.gender is not None and str(payload.gender).strip():
        after["gender"] = str(payload.gender).strip()

    if payload.symptoms is not None:
        if isinstance(payload.symptoms, list):
            symptoms_text = ", ".join([str(s).strip() for s in payload.symptoms if str(s).strip()])
        else:
            symptoms_text = str(payload.symptoms).strip()
        after["symptoms"] = symptoms_text
        after["symptom_count"] = len(split_symptoms_text(symptoms_text))

    if payload.height_cm is not None:
        after["height_cm"] = float(payload.height_cm)
    if payload.weight_kg is not None:
        after["weight_kg"] = float(payload.weight_kg)
    if payload.bmi is not None:
        after["bmi"] = float(payload.bmi)

    if payload.glucose is not None:
        after["glucose"] = float(payload.glucose)
    if payload.glucose_type is not None:
        after["glucose_type"] = str(payload.glucose_type)

    if payload.systolic_bp is not None and payload.diastolic_bp is not None:
        systolic = float(payload.systolic_bp)
        diastolic = float(payload.diastolic_bp)
        after["blood_pressure"] = f"{int(round(systolic))}/{int(round(diastolic))}"

    if "submitted_at" not in after or not str(after.get("submitted_at") or "").strip():
        after["submitted_at"] = changed_at
    after["updated_at"] = changed_at

    update_user_health_details(patient_email, after)

    changes = compute_health_details_changes(before, after)
    event: dict[str, Any] | None = None
    if changes:
        try:
            event = {
                "patient_email": patient_email,
                "changed_at": changed_at,
                "changed_by_email": updated_by_email,
                "changed_by_role": updated_by_role,
                "source": "clinical_update",
                "changes": changes,
                "before": snapshot_health_details(before),
                "after": snapshot_health_details(after),
            }
            append_patient_record_history(event)
        except Exception:
            event = None

    return {
        "message": "Patient record updated.",
        "patient_email": patient_email,
        "changed_at": changed_at,
        "changed_fields": [c["field"] for c in changes],
        "event": event,
        "health_details": after,
    }
