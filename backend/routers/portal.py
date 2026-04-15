from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, RedirectResponse, Response

from backend.services.shared import (
    FAVICON_ICO_PATH,
    FAVICON_SVG_PATH,
    HTML_PAGES,
    get_session_user,
    html_page,
)

router = APIRouter()


@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    if FAVICON_ICO_PATH.exists():
        return FileResponse(path=str(FAVICON_ICO_PATH))

    if FAVICON_SVG_PATH.exists():
        return RedirectResponse(url="/static/img/heartbeat.svg")

    return Response(status_code=204)


@router.get("/")
def root() -> FileResponse:
    return html_page(HTML_PAGES["root"])


@router.get("/about")
def about_page() -> FileResponse:
    return html_page(HTML_PAGES["about"])


@router.get("/patient")
def patient_dashboard() -> FileResponse:
    return html_page(HTML_PAGES["patient"])


@router.get("/patient/risk-analysis")
def patient_risk_analysis_dashboard() -> FileResponse:
    # Uses the same UI as /patient, but provides a dedicated URL for the risk view.
    return html_page(HTML_PAGES["patient"])


@router.get("/patient/history")
def patient_history_dashboard() -> FileResponse:
    return html_page(HTML_PAGES["patient_history"])


@router.get("/patient/appointments")
def patient_appointments_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "patient":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["patient_appointments"])


@router.get("/patient/book")
def patient_book_appointment(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "patient":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["patient_book"])


@router.get("/patient/risk")
def patient_risk_dashboard() -> RedirectResponse:
    return RedirectResponse(url="/patient/risk-analysis")


@router.get("/nurse")
def nurse_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "nurse":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["nurse"])


@router.get("/nurse/editor")
def nurse_record_editor(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "nurse":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["nurse_editor"])


@router.get("/doctor")
def doctor_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor"])


@router.get("/doctor/past")
def doctor_past_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor_past"])


@router.get("/doctor/patients")
def doctor_patients_page(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor_patients"])


@router.get("/doctor/leave")
def doctor_leave_page(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor_leave"])


@router.get("/login")
def login_page() -> FileResponse:
    return html_page(HTML_PAGES["login"])


@router.get("/register")
def register_page() -> FileResponse:
    return html_page(HTML_PAGES["register"])


@router.get("/health-details")
def health_details_page(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "patient":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["health_details"])


@router.get("/predict-risk")
def predict_risk_form() -> RedirectResponse:
    return RedirectResponse(url="/nurse/editor")
