from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from backend.services.shared import (
    AppointmentCancelRequest,
    AppointmentCreateRequest,
    AppointmentResponse,
    DoctorLeaveRequest,
    appointment_to_payload,
    dbmod,
    doctor_is_on_leave,
    doctor_leave_for,
    emergency_doctors_available,
    get_all_users,
    get_db,
    get_user_by_email,
    load_doctors_dataset,
    load_users,
    normalize_email,
    normalize_phone,
    nurse_rule_assessment_from_health_details,
    parse_iso_datetime,
    require_role,
    require_session,
    safe_normalize_role,
    user_directory_payload,
    validate_email,
)

router = APIRouter()


@router.get("/portal/directory")
def portal_directory(
    request: Request,
    scheduled_for: str | None = Query(default=None, min_length=10, max_length=64),
) -> dict[str, Any]:
    session_user = require_session(request)
    role = str(session_user.get("role") or "")

    when: datetime | None = None
    if scheduled_for and str(scheduled_for).strip():
        when = parse_iso_datetime(str(scheduled_for).strip())

    users = get_all_users()
    patients = [user_directory_payload(user) for user in users if user.get("role") == "patient"]
    nurses = [user_directory_payload(user) for user in users if user.get("role") == "nurse"]
    doctors_from_users = [user_directory_payload(user) for user in users if user.get("role") == "doctor"]

    doctors: list[dict[str, Any]] = []
    # CHANGE (patient "Select doctor" dropdown fix):
    # The dropdown must reflect real doctor accounts in SQLite (users.role == "doctor").
    # Previously, we always appended demo/static entries from datasets/doctors.csv, which
    # caused old doctors to keep showing even after real doctors registered.
    #
    # We still *enrich* doctor accounts with optional dataset metadata (specialization /
    # emergency_doctor / available_time) when an email matches, but we never add dataset-
    # only doctors to the directory list.
    dataset_by_email: dict[str, dict[str, Any]] = {}
    try:
        for d in load_doctors_dataset():
            key = normalize_email((d.get("doctor_id") or "").strip())
            if key:
                dataset_by_email[key] = d
    except Exception:
        dataset_by_email = {}

    seen: set[str] = set()
    for doc in doctors_from_users:
        key = normalize_email(doc.get("email") or "")
        if not key or key in seen:
            continue
        seen.add(key)

        # Optional enrichment for display only (does not affect authorization).
        dataset_doc = dataset_by_email.get(key) or {}
        doc = {
            **doc,
            "specialization": (dataset_doc.get("specialization") or "").strip(),
            "available_time": (dataset_doc.get("available_time") or "").strip(),
            "emergency_doctor": bool(dataset_doc.get("emergency_doctor")),
        }
        if when is not None:
            leave = doctor_leave_for(key, when)
            doc = {**doc, "on_leave": bool(leave), "leave": leave, "available": not bool(leave)}
        doctors.append(doc)

    return {
        "scheduled_for": when.isoformat() if when is not None else None,
        "patients": patients if role in {"nurse", "doctor"} else [],
        "nurses": nurses if role in {"nurse", "doctor"} else [],
        "doctors": doctors,
    }


@router.get("/api/doctor/leave/check")
def doctor_leave_check(
    request: Request,
    doctor_email: str = Query(..., min_length=5, max_length=254),
    scheduled_for: str = Query(..., min_length=10, max_length=64),
) -> dict[str, Any]:
    require_role("patient", "nurse", "doctor")(require_session(request))
    when = parse_iso_datetime(scheduled_for)
    doctor_id = validate_email(doctor_email)
    leave = doctor_leave_for(doctor_id, when)
    emergency = emergency_doctors_available(when)
    if not leave:
        return {"on_leave": False, "message": "Doctor is available.", "emergency_doctors": emergency}

    return {
        "on_leave": True,
        "leave": leave,
        "emergency_doctors": emergency,
        "message": "Doctor is on leave for the selected date.",
    }


@router.get("/api/doctor/leaves")
def doctor_leaves(request: Request, doctor_email: str = Query(..., min_length=5, max_length=254)) -> list[dict[str, Any]]:
    session_user = require_role("doctor")(require_session(request))
    if normalize_email(str(session_user.get("email") or "")) != validate_email(doctor_email):
        raise HTTPException(status_code=403, detail="Doctors can only access their own leave calendar.")
    leaves = dbmod.list_doctor_leaves(get_db(), normalize_email(doctor_email))
    leaves.sort(key=lambda l: l.get("start_at") or "", reverse=True)
    return leaves


@router.post("/api/doctor/leaves")
def upsert_doctor_leave(request: Request, payload: DoctorLeaveRequest) -> dict[str, Any]:
    session_user = require_role("doctor")(require_session(request))
    if normalize_email(str(session_user.get("email") or "")) != validate_email(payload.doctor_email):
        raise HTTPException(status_code=403, detail="Doctors can only update their own leave calendar.")
    doctor_id = validate_email(payload.doctor_email)
    start_at = parse_iso_datetime(payload.start_at)
    end_at = parse_iso_datetime(payload.end_at)
    if end_at < start_at:
        raise HTTPException(status_code=400, detail="Leave end time must be after start time.")

    created_at = datetime.now(timezone.utc).isoformat()
    leave = dbmod.insert_doctor_leave(
        get_db(),
        {
            "doctor_email": doctor_id,
            "start_at": start_at.isoformat(),
            "end_at": end_at.isoformat(),
            "reason": (payload.reason or "").strip() or None,
            "created_at": created_at,
            "updated_at": created_at,
        },
    )

    return {"message": "Leave saved.", "leave": leave}


@router.post("/appointments/book", response_model=AppointmentResponse)
def book_appointment(request: Request, payload: AppointmentCreateRequest) -> AppointmentResponse:
    session_user = require_role("patient")(require_session(request))
    patient_email = validate_email(payload.patient_email)
    if normalize_email(str(session_user.get("email") or "")) != patient_email:
        raise HTTPException(status_code=403, detail="You can only book appointments for your own account.")
    patient = get_user_by_email(patient_email)
    if not patient or patient.get("role") != "patient":
        raise HTTPException(status_code=404, detail="Patient account not found.")

    scheduled_for_dt = parse_iso_datetime(payload.scheduled_for)
    created_at = datetime.now(timezone.utc).isoformat()
    reason = (payload.reason or payload.appointment_type).strip()

    if payload.doctor_email:
        doctor_id = validate_email(payload.doctor_email)
        if doctor_is_on_leave(doctor_id, scheduled_for_dt):
            emergency = emergency_doctors_available(scheduled_for_dt)
            emergency_text = ", ".join([d.get("doctor_name") or d.get("doctor_id") for d in emergency[:3]])
            hint = f" Emergency doctors: {emergency_text}." if emergency_text else ""
            raise HTTPException(status_code=409, detail=f"Selected doctor is on leave for that date.{hint}")

    contact_phone = normalize_phone(payload.contact_info)

    appointment = dbmod.insert_appointment(
        get_db(),
        {
            "patient_email": patient_email,
            "scheduled_for": scheduled_for_dt.isoformat(),
            "status": "Pending",
            "reason": reason,
            "appointment_type": payload.appointment_type.strip(),
            "doctor_email": doctor_id if payload.doctor_email else None,
            "contact_info": contact_phone,
            "priority": None,
            "notes": payload.notes.strip() if payload.notes else None,
            "created_at": created_at,
        },
    )

    return AppointmentResponse(**appointment_to_payload(appointment))


@router.get("/appointments", response_model=list[AppointmentResponse])
def list_appointments(
    request: Request,
    patient_email: str = Query(..., min_length=5, max_length=254),
) -> list[AppointmentResponse]:
    session_user = require_role("patient")(require_session(request))
    email = validate_email(patient_email)
    if normalize_email(str(session_user.get("email") or "")) != email:
        raise HTTPException(status_code=403, detail="You can only view your own appointments.")
    patient = get_user_by_email(email)
    if not patient or patient.get("role") != "patient":
        raise HTTPException(status_code=404, detail="Patient account not found.")

    appointments = dbmod.list_appointments_for_patient(get_db(), email)
    return [AppointmentResponse(**appointment_to_payload(appointment)) for appointment in appointments]


@router.post("/appointments/cancel", response_model=AppointmentResponse)
def cancel_appointment(request: Request, payload: AppointmentCancelRequest) -> AppointmentResponse:
    session_user = require_role("patient")(require_session(request))
    email = validate_email(payload.patient_email)
    if normalize_email(str(session_user.get("email") or "")) != email:
        raise HTTPException(status_code=403, detail="You can only cancel your own appointments.")
    patient = get_user_by_email(email)
    if not patient or patient.get("role") != "patient":
        raise HTTPException(status_code=404, detail="Patient account not found.")

    now = datetime.now(timezone.utc)
    match = dbmod.get_appointment(get_db(), int(payload.appointment_id))
    if not match or normalize_email(match.get("patient_email") or "") != email:
        raise HTTPException(status_code=404, detail="Appointment not found.")

    status = str(match.get("status") or "Pending")
    if status.lower() in {"cancelled", "canceled", "completed"}:
        raise HTTPException(status_code=409, detail="This appointment cannot be cancelled.")

    scheduled = parse_iso_datetime(match.get("scheduled_for") or "")
    if scheduled < now:
        raise HTTPException(status_code=409, detail="Past appointments cannot be cancelled.")

    notes_out = None
    if payload.cancel_reason and payload.cancel_reason.strip():
        existing = (match.get("notes") or "").strip()
        prefix = f"Cancelled: {payload.cancel_reason.strip()}"
        notes_out = (prefix if not existing else f"{existing}\n{prefix}")[:1000]

    dbmod.update_appointment_status(get_db(), int(payload.appointment_id), "Cancelled", notes_out)
    match = dbmod.get_appointment(get_db(), int(payload.appointment_id)) or match
    return AppointmentResponse(**appointment_to_payload(match))


@router.get("/appointments/dashboard")
def appointments_dashboard(request: Request) -> dict[str, Any]:
    session_user = require_role("nurse", "doctor")(require_session(request))
    now = datetime.now(timezone.utc)

    # CHANGE (doctor dashboard privacy):
    # Nurses see the global dashboard (all appointments), but doctors must only see
    # appointments assigned to their own account (doctor_email == session user email).
    role = str(session_user.get("role") or "").strip().lower()
    if role == "doctor":
        doctor_email = normalize_email(str(session_user.get("email") or ""))
        appointments = dbmod.list_appointments_for_doctor(get_db(), doctor_email)
    else:
        appointments = dbmod.list_all_appointments(get_db())

    upcoming: list[dict[str, Any]] = []
    past: list[dict[str, Any]] = []
    for appointment in appointments:
        scheduled = parse_iso_datetime(appointment["scheduled_for"])
        payload = appointment_to_payload(appointment)
        patient = get_user_by_email(payload.get("patient_email") or "")
        triage = nurse_rule_assessment_from_health_details(patient.get("health_details") if patient else None)
        if triage:
            payload["nurse_triage"] = triage
        if scheduled >= now and payload["status"] in ("Pending", "Confirmed"):
            upcoming.append(payload)
        else:
            past.append(payload)

    rank = {"Critical": 0, "High": 1, "Moderate": 2, "Low": 3, "Needs vitals": 4}

    def triage_sort_key(item: dict[str, Any]) -> tuple[int, int, datetime]:
        triage_item = item.get("nurse_triage") or {}
        level = str(triage_item.get("triage_level") or "")
        show_on_top = bool(triage_item.get("show_on_top"))
        scheduled_for = parse_iso_datetime(item.get("scheduled_for") or datetime.now(timezone.utc).isoformat())
        return (0 if show_on_top else 1, rank.get(level, 9), scheduled_for)

    upcoming.sort(key=triage_sort_key)

    stats = {
        "critical": 0,
        "high": 0,
        "moderate": 0,
        "low": 0,
        "needs_vitals": 0,
    }
    for item in upcoming:
        triage_item = item.get("nurse_triage") or {}
        level = str(triage_item.get("triage_level") or "")
        if level == "Critical":
            stats["critical"] += 1
        elif level == "High":
            stats["high"] += 1
        elif level == "Moderate":
            stats["moderate"] += 1
        elif level == "Low":
            stats["low"] += 1
        elif level == "Needs vitals":
            stats["needs_vitals"] += 1

    return {"upcoming": upcoming, "past": past, "stats": stats}


@router.get("/api/doctor/patients")
def doctor_patients_directory(request: Request) -> list[dict[str, Any]]:
    require_role("doctor")(require_session(request))
    users = load_users()
    patients = [user for user in users if safe_normalize_role(user.get("role") or "patient") == "patient"]
    patients.sort(key=lambda entry: (entry.get("full_name") or "").strip().lower())
    return [user_directory_payload(patient) for patient in patients]
