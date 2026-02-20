from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from flask_app import app as flask_portal_app

from doctor_auth import DoctorAuthManager, DoctorLoginRequest, DoctorSignupRequest
from patient_auth import (
    PatientAuthManager,
    PatientLoginRequest,
    PatientSignupRequest,
)
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


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(
    title="Disease Risk Prediction Engine",
    version="1.0.0",
    description="Live disease risk estimation integrated with appointment booking.",
)

try:
    risk_engine = RiskEngine(MODEL_PATH, LABEL_ENCODER_PATH)
except Exception as exc:  # pragma: no cover - startup guard
    raise RuntimeError(f"Failed to initialize risk engine: {exc}") from exc

APPOINTMENTS: List[Dict[str, Any]] = []
patient_auth_manager = PatientAuthManager()
doctor_auth_manager = DoctorAuthManager()


def _read_template(name: str) -> str:
    path = TEMPLATES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/portal/login", status_code=307)


@app.get("/login", response_class=HTMLResponse)
def login_page() -> RedirectResponse:
    return RedirectResponse(url="/portal/login", status_code=307)


@app.get("/home", response_class=HTMLResponse)
def home_page() -> RedirectResponse:
    return RedirectResponse(url="/portal/login", status_code=307)


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
        return {"status": "ok", "message": "Login successful. Redirecting to Patient Portal."}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
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
        doctor_auth_manager.login(payload.doctor_id, payload.password)
        return {"status": "ok", "message": "Login successful. Redirecting to Doctor Dashboard."}
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Login failed: {exc}") from exc


@app.get("/doctor", response_class=HTMLResponse)
def doctor_page() -> str:
    return _read_template("doctor.html")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "risk_prediction_engine"}


@app.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> RiskPredictionResponse:
    try:
        return risk_engine.predict(payload.patient_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/book-appointment", response_model=AppointmentBookingResponse)
def book_appointment(payload: AppointmentBookingRequest) -> AppointmentBookingResponse:
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
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {exc}") from exc

    booking_status = "confirmed"
    response = AppointmentBookingResponse(
        booking_status=booking_status,
        patient_id=payload.patient_id,
        doctor_id=doctor_id,
        appointment_time=payload.appointment_time,
        risk_assessment=risk_assessment,
    )
    APPOINTMENTS.append(
        {
            "booked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "patient_id": payload.patient_id,
            "doctor_id": doctor_id,
            "appointment_time": payload.appointment_time.isoformat(),
            "patient_features": to_jsonable(features),
            "risk_assessment": response.risk_assessment.model_dump(),
        }
    )
    return response


@app.get("/patient-features/{patient_id}")
def patient_features(patient_id: str) -> Dict[str, Any]:
    try:
        features = risk_engine.get_patient_features(patient_id)
        return {"patient_id": patient_id, "patient_features": features}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/appointments")
def list_appointments() -> Dict[str, Any]:
    return {"appointments": APPOINTMENTS}


# Portal entry from main.py: all patient/doctor UI flows live in flask_app.py
app.mount("/portal", WSGIMiddleware(flask_portal_app))


# Compatibility routes so previous URLs continue working without /portal prefix.
@app.api_route("/patient/login", methods=["GET", "POST"], include_in_schema=False)
def patient_login_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/login", status_code=307)


@app.api_route("/patient/signup", methods=["GET", "POST"], include_in_schema=False)
def patient_signup_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/signup", status_code=307)


@app.api_route("/patient/health-details", methods=["GET", "POST"], include_in_schema=False)
def patient_health_details_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/health-details", status_code=307)


@app.get("/patient/health-confirmation", include_in_schema=False)
def patient_health_confirmation_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/health-confirmation", status_code=307)


@app.get("/patient/book-appointment", include_in_schema=False)
def patient_book_appointment_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/book-appointment", status_code=307)


@app.get("/patient/features/{patient_id}", include_in_schema=False)
def patient_features_portal(patient_id: str) -> RedirectResponse:
    return RedirectResponse(url=f"/portal/patient/features/{patient_id}", status_code=307)


@app.post("/patient/book-appointment-submit", include_in_schema=False)
def patient_book_appointment_submit_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/patient/book-appointment-submit", status_code=307)


@app.api_route("/doctor/login", methods=["GET", "POST"], include_in_schema=False)
def doctor_login_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/login", status_code=307)


@app.api_route("/doctor/signup", methods=["GET", "POST"], include_in_schema=False)
def doctor_signup_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/signup", status_code=307)


@app.get("/doctor/dashboard", include_in_schema=False)
def doctor_dashboard_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/dashboard", status_code=307)


@app.get("/doctor/appointments", include_in_schema=False)
def doctor_appointments_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/appointments", status_code=307)


@app.post("/doctor/predict-risk", include_in_schema=False)
def doctor_predict_risk_portal() -> RedirectResponse:
    return RedirectResponse(url="/portal/doctor/predict-risk", status_code=307)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

