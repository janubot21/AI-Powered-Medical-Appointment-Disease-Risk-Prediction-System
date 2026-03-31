from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from flask import jsonify, redirect, session, url_for


def _is_patient_authenticated() -> bool:
    return bool(str(session.get("patient_id", "")).strip()) and bool(session.get("patient_authenticated"))


def _is_doctor_authenticated() -> bool:
    return bool(str(session.get("doctor_id", "")).strip()) and bool(session.get("doctor_authenticated"))


def _is_nurse_authenticated() -> bool:
    return bool(str(session.get("nurse_id", "")).strip()) and bool(session.get("nurse_authenticated"))


def _active_role() -> str:
    if _is_doctor_authenticated():
        return "doctor"
    if _is_nurse_authenticated():
        return "nurse"
    if _is_patient_authenticated():
        return "patient"
    return "anonymous"


def patient_required(*, api: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            role = _active_role()
            if role == "patient":
                return func(*args, **kwargs)
            if role == "doctor":
                if api:
                    return jsonify({"detail": "Forbidden: doctor account cannot access patient endpoint."}), 403
                return redirect(url_for("doctor_dashboard"))
            if role == "nurse":
                if api:
                    return jsonify({"detail": "Forbidden: nurse account cannot access patient endpoint."}), 403
                return redirect(url_for("nurse_dashboard"))
            if api:
                return jsonify({"detail": "Unauthorized"}), 401
            return redirect(url_for("patient_login"))

        return wrapper

    return decorator


def doctor_required(*, api: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            role = _active_role()
            if role == "doctor":
                return func(*args, **kwargs)
            if role == "nurse":
                if api:
                    return jsonify({"detail": "Forbidden: nurse account cannot access doctor endpoint."}), 403
                return redirect(url_for("nurse_dashboard"))
            if role == "patient":
                if api:
                    return jsonify({"detail": "Forbidden: patient account cannot access doctor endpoint."}), 403
                return redirect(url_for("book_appointment"))
            if api:
                return jsonify({"detail": "Unauthorized"}), 401
            return redirect(url_for("doctor_login"))

        return wrapper

    return decorator


def nurse_required(*, api: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            role = _active_role()
            if role == "nurse":
                return func(*args, **kwargs)
            if role == "doctor":
                if api:
                    return jsonify({"detail": "Forbidden: doctor account cannot access nurse endpoint."}), 403
                return redirect(url_for("doctor_dashboard"))
            if role == "patient":
                if api:
                    return jsonify({"detail": "Forbidden: patient account cannot access nurse endpoint."}), 403
                return redirect(url_for("book_appointment"))
            if api:
                return jsonify({"detail": "Unauthorized"}), 401
            return redirect(url_for("nurse_login"))

        return wrapper

    return decorator


def _clear_patient_session() -> None:
    session.pop("patient_id", None)
    session.pop("patient_name", None)
    session.pop("patient_authenticated", None)
    session.pop("allow_health_details", None)
    session.pop("health_confirmation", None)


def _clear_doctor_session() -> None:
    session.pop("doctor_id", None)
    session.pop("doctor_name", None)
    session.pop("doctor_authenticated", None)


def _clear_nurse_session() -> None:
    session.pop("nurse_id", None)
    session.pop("nurse_name", None)
    session.pop("nurse_authenticated", None)
