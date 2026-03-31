from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from flask_app import app as flask_portal_app

from api_auth import (
    APIPrincipal,
    assert_doctor_scope,
    assert_patient_scope,
    create_api_token,
    require_api_roles,
)
from doctor_auth import DoctorAuthManager, DoctorLoginRequest, DoctorSignupRequest 
from nurse_auth import NurseAuthManager, NurseLoginRequest, NurseSignupRequest
from paths import PATIENT_DB_PATH
from patient_auth import (
    PatientAuthManager,
    PatientLoginRequest,
    PatientSignupRequest,
)   
from patient_db import PatientDatabase
from predict import (
    AppointmentBookingRequest,    
    AppointmentBookingResponse,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    RiskEngine,
    RiskPredictionRequest,
    RiskPredictionResponse,
    default_features_json, 
    to_jsonable,
) 
from triage import determine_priority


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
TEMPLATES_DIR = FRONTEND_DIR / "templates"
STATIC_DIR = FRONTEND_DIR / "static"
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disease Risk Prediction Engine",
    version="1.0.0",
    description="Live disease risk estimation integrated with appointment booking.",
)

# Serve shared frontend assets (login background, images, etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

try:
    risk_engine = RiskEngine(MODEL_PATH, LABEL_ENCODER_PATH)
except Exception as exc:  # pragma: no cover - startup guard
    raise RuntimeError(f"Failed to initialize risk engine: {exc}") from exc

patient_auth_manager = PatientAuthManager()
doctor_auth_manager = DoctorAuthManager()
nurse_auth_manager = NurseAuthManager()
patient_db = PatientDatabase(PATIENT_DB_PATH)


def _read_template(name: str) -> str:
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


def _doctor_can_access_patient(doctor_id: str, patient_id: str) -> bool:
    return bool(patient_db.list_appointments(patient_id=patient_id, doctor_id=doctor_id))


@app.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/portal/login", status_code=302)


@app.get("/login", response_class=HTMLResponse)
def login_page() -> RedirectResponse:
    return RedirectResponse(url="/portal/login", status_code=302)


@app.get("/home", response_class=HTMLResponse)
def home_page() -> RedirectResponse:
    return RedirectResponse(url="/portal/login", status_code=302)


@app.get("/about", response_class=HTMLResponse)
def about_page() -> RedirectResponse:
    return RedirectResponse(url="/portal/about", status_code=302)


@app.get("/patient", response_class=HTMLResponse)
def patient_page() -> str:
    html = _read_template("patient.html")
    return html.replace("__DEFAULT_FEATURES_JSON__", default_features_json(risk_engine))


@app.get("/patient-auth", response_class=HTMLResponse)
def patient_auth_page() -> str:
    return _read_template("patient_auth.html")


@app.post("/patient-signup")
def patient_signup(payload: PatientSignupRequest) -> Dict[str, str]:
    try:
        patient_auth_manager.signup(payload.patient_id, payload.password)
        return {"status": "ok", "message": "Signup successful. Redirecting to Patient Portal."}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Signup failed: {exc}") from exc


@app.post("/patient-login")
def patient_login(payload: PatientLoginRequest) -> Dict[str, str]:
    try:
        patient_auth_manager.login(payload.patient_id, payload.password)
        return {
            "status": "ok",
            "message": "Login successful. Redirecting to Patient Portal.",
            "patient_id": payload.patient_id,
            "role": "patient",
            "api_token": create_api_token("patient", payload.patient_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Patient API login failed for patient_id=%s", payload.patient_id)
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc


@app.get("/doctor-auth", response_class=HTMLResponse)
def doctor_auth_page() -> str:
    return _read_template("doctor_auth.html")


@app.post("/doctor-signup")
def doctor_signup(payload: DoctorSignupRequest) -> Dict[str, str]:
    try:
        doctor_auth_manager.signup(payload.doctor_id, payload.password)
        return {"status": "ok", "message": "Signup successful. Redirecting to Doctor Dashboard."}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Signup failed: {exc}") from exc


@app.post("/doctor-login")
def doctor_login(payload: DoctorLoginRequest) -> Dict[str, str]:
    try:
        resolved_doctor_id = doctor_auth_manager.login(
            payload.doctor_id,
            payload.id_type or "",
            payload.id_number or "",
            payload.password,
        )
        return {
            "status": "ok",
            "message": "Login successful. Redirecting to Doctor Dashboard.",
            "doctor_id": resolved_doctor_id,
            "role": "doctor",
            "api_token": create_api_token("doctor", resolved_doctor_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Doctor API login failed for doctor_id=%s", payload.doctor_id)
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc


@app.post("/nurse-signup")
def nurse_signup(payload: NurseSignupRequest) -> Dict[str, str]:
    try:
        nurse_auth_manager.signup(payload.nurse_id, payload.password)
        return {"status": "ok", "message": "Signup successful. Redirecting to Nurse Dashboard."}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Nurse API signup failed for nurse_id=%s", payload.nurse_id)
        raise HTTPException(status_code=500, detail=f"Signup failed: {exc}") from exc


@app.post("/nurse-login")
def nurse_login(payload: NurseLoginRequest) -> Dict[str, str]:
    try:
        resolved_nurse_id = nurse_auth_manager.login(payload.nurse_id, payload.password)
        return {
            "status": "ok",
            "message": "Login successful. Redirecting to Nurse Dashboard.",
            "nurse_id": resolved_nurse_id,
            "role": "nurse",
            "api_token": create_api_token("nurse", resolved_nurse_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Nurse API login failed for nurse_id=%s", payload.nurse_id)
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc


@app.get("/doctor", response_class=HTMLResponse)
def doctor_page() -> str:
    return _read_template("doctor.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "risk_prediction_engine"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(path=str(STATIC_DIR / "favicon.svg"), media_type="image/svg+xml")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def chrome_devtools_probe() -> Response:
    return Response(status_code=204)


@app.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(
    payload: RiskPredictionRequest,
    _: APIPrincipal = Depends(require_api_roles("patient", "doctor", "nurse")),
) -> RiskPredictionResponse:
    try:
        return risk_engine.predict(payload.patient_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Risk prediction failed for authenticated API request.")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/book-appointment", response_model=AppointmentBookingResponse)
def book_appointment(
    payload: AppointmentBookingRequest,
    principal: APIPrincipal = Depends(require_api_roles("patient", "doctor", "nurse")),
) -> AppointmentBookingResponse:
    if principal.role == "patient":
        assert_patient_scope(principal, payload.patient_id)
    if principal.role == "doctor":
        target_doctor_id = (payload.doctor_id or "").strip() or principal.subject
        assert_doctor_scope(principal, target_doctor_id)

    features = payload.patient_features
    doctor_id = (payload.doctor_id or "").strip() or "Unassigned"
    if features is None:
        try:
            features = risk_engine.get_patient_features(payload.patient_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        risk_assessment = risk_engine.predict(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Risk assessment failed during booking for patient_id=%s", payload.patient_id)
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {exc}") from exc

    booking_status = "confirmed"
    priority = determine_priority(risk_assessment.risk_level)
    response = AppointmentBookingResponse(
        booking_status=booking_status,
        patient_id=payload.patient_id,
        doctor_id=doctor_id,
        appointment_time=payload.appointment_time,
        risk_assessment=risk_assessment,
        appointment_priority=priority.priority,
        recommended_slot=priority.recommended_slot,
    )
    patient_profile = patient_db.get_profile(payload.patient_id)
    patient_db.add_appointment(
        {
            "appointment_id": uuid4().hex,
            "booked_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "patient_id": payload.patient_id,
            "patient_name": patient_profile.get("patient_name") or "",
            "contact_info": "",
            "doctor_id": doctor_id,
            "appointment_type": "API Booking",
            "appointment_time": payload.appointment_time.isoformat(),
            "patient_features": to_jsonable(features),
            "risk_assessment": response.risk_assessment.model_dump(),
            "appointment_priority": response.appointment_priority,
            "recommended_slot": response.recommended_slot,
            "priority_badge_text": priority.badge_text,
            "priority_badge_color": priority.badge_color,
        }
    )
    return response


@app.get("/patient-features/{patient_id}")
def patient_features(
    patient_id: str,
    principal: APIPrincipal = Depends(require_api_roles("patient", "doctor", "nurse")),
) -> Dict[str, Any]:
    if principal.role == "patient":
        assert_patient_scope(principal, patient_id)
    elif principal.role == "doctor" and not _doctor_can_access_patient(principal.subject, patient_id):
        raise HTTPException(status_code=403, detail="Forbidden for this doctor's patients.")
    try:
        features = risk_engine.get_patient_features(patient_id)
        return {"patient_id": patient_id, "patient_features": features}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Patient feature lookup failed for patient_id=%s", patient_id)
        raise HTTPException(status_code=500, detail=f"Patient lookup failed: {exc}") from exc


@app.get("/appointments")
def list_appointments(
    patient_id: str | None = None,
    doctor_id: str | None = None,
    principal: APIPrincipal = Depends(require_api_roles("patient", "doctor", "nurse")),
) -> Dict[str, Any]:
    requested_patient_id = str(patient_id or "").strip()
    requested_doctor_id = str(doctor_id or "").strip()

    if principal.role == "patient":
        target_patient_id = requested_patient_id or principal.subject
        assert_patient_scope(principal, target_patient_id)
        requested_patient_id = target_patient_id
    elif principal.role == "doctor":
        target_doctor_id = requested_doctor_id or principal.subject
        assert_doctor_scope(principal, target_doctor_id)
        requested_doctor_id = target_doctor_id
        if requested_patient_id and not _doctor_can_access_patient(principal.subject, requested_patient_id):
            raise HTTPException(status_code=403, detail="Forbidden for this doctor's patients.")

    return {
        "appointments": patient_db.list_appointments(
            patient_id=requested_patient_id or None,
            doctor_id=requested_doctor_id or None,
        )
    }


# Portal entry from main.py: all patient/doctor UI flows live in flask_app.py
app.mount("/portal", WSGIMiddleware(flask_portal_app))


# Compatibility routes so previous URLs continue working without /portal prefix.
@app.get("/patient/login", include_in_schema=False)
def patient_login_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/login", status_code=302)


@app.post("/patient/login", include_in_schema=False)
def patient_login_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/login", status_code=307)


@app.get("/patient/signup", include_in_schema=False)
def patient_signup_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/signup", status_code=302)


@app.post("/patient/signup", include_in_schema=False)
def patient_signup_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/signup", status_code=307)


@app.get("/patient/health-details", include_in_schema=False)
def patient_health_details_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/health-details", status_code=302)


@app.post("/patient/health-details", include_in_schema=False)
def patient_health_details_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/health-details", status_code=307)


@app.get("/patient/health-confirmation", include_in_schema=False)
def patient_health_confirmation_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/health-confirmation", status_code=302)


@app.get("/patient/book-appointment", include_in_schema=False)
def patient_book_appointment_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/book-appointment", status_code=302)


@app.get("/patient/booking-confirmation", include_in_schema=False)
def patient_booking_confirmation_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/booking-confirmation", status_code=302)


@app.get("/patient/ai-chat", include_in_schema=False)
def patient_ai_chat_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/ai-chat", status_code=302)


@app.post("/patient/ai-chat/message", include_in_schema=False)
def patient_ai_chat_message_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/ai-chat/message", status_code=307)


@app.get("/patient/logout", include_in_schema=False)
def patient_logout_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/logout", status_code=302)


@app.get("/patient/features/{patient_id}", include_in_schema=False)
def patient_features_portal(patient_id: str) -> RedirectResponse:
    return RedirectResponse(url=f"/portal/patient/features/{patient_id}", status_code=302)


@app.post("/patient/features/{patient_id}", include_in_schema=False)
def patient_features_portal_post(patient_id: str) -> RedirectResponse:
    return RedirectResponse(url=f"/portal/patient/features/{patient_id}", status_code=307)


@app.post("/patient/book-appointment-submit", include_in_schema=False)
def patient_book_appointment_submit_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/book-appointment-submit", status_code=307)


@app.post("/patient/predict-risk", include_in_schema=False)
def patient_predict_risk_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/predict-risk", status_code=307)


@app.get("/doctor/login", include_in_schema=False)
def doctor_login_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/login", status_code=302)


@app.post("/doctor/login", include_in_schema=False)
def doctor_login_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/login", status_code=307)


@app.get("/doctor/signup", include_in_schema=False)
def doctor_signup_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/signup", status_code=302)


@app.post("/doctor/signup", include_in_schema=False)
def doctor_signup_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/signup", status_code=307)


@app.get("/doctor/dashboard", include_in_schema=False)
def doctor_dashboard_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/dashboard", status_code=302)


@app.get("/doctor/logout", include_in_schema=False)
def doctor_logout_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/logout", status_code=302)


@app.get("/doctor/appointments", include_in_schema=False)
def doctor_appointments_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/appointments", status_code=302)


@app.get("/doctor/patient-database", include_in_schema=False)
def doctor_patient_database_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/patient-database", status_code=302)


@app.post("/doctor/predict-risk", include_in_schema=False)
def doctor_predict_risk_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/predict-risk", status_code=307)


@app.get("/nurse/login", include_in_schema=False)
def nurse_login_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/login", status_code=302)


@app.post("/nurse/login", include_in_schema=False)
def nurse_login_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/login", status_code=307)


@app.get("/nurse/signup", include_in_schema=False)
def nurse_signup_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/signup", status_code=302)


@app.post("/nurse/signup", include_in_schema=False)
def nurse_signup_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/signup", status_code=307)


@app.get("/nurse/dashboard", include_in_schema=False)
def nurse_dashboard_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/dashboard", status_code=302)


@app.get("/nurse/appointment-queue", include_in_schema=False)
def nurse_queue_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/appointment-queue", status_code=302)


@app.get("/nurse/patient-records", include_in_schema=False)
def nurse_patient_records_portal_get() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/patient-records", status_code=302)


@app.post("/nurse/patient-records", include_in_schema=False)
def nurse_patient_records_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/patient-records", status_code=307)


@app.post("/nurse/predict-risk", include_in_schema=False)
def nurse_predict_risk_portal_post() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/predict-risk", status_code=307)


@app.get("/nurse/logout", include_in_schema=False)
def nurse_logout_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/nurse/logout", status_code=302)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(BASE_DIR))
  
