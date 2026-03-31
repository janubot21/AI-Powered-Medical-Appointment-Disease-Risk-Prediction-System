from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app_config import get_flask_secret_key


TOKEN_MAX_AGE_SECONDS = 8 * 60 * 60
_bearer = HTTPBearer(auto_error=False)
_serializer = URLSafeTimedSerializer(get_flask_secret_key(), salt="medical-api-auth")


@dataclass(frozen=True)
class APIPrincipal:
    role: str
    subject: str


def create_api_token(role: str, subject: str) -> str:
    normalized_role = str(role).strip().lower()
    normalized_subject = str(subject).strip()
    if normalized_role not in {"patient", "doctor", "nurse"}:
        raise ValueError("Unsupported API auth role.")
    if not normalized_subject:
        raise ValueError("API auth subject cannot be empty.")
    return _serializer.dumps({"role": normalized_role, "subject": normalized_subject})


def _decode_api_token(token: str) -> APIPrincipal:
    try:
        payload = _serializer.loads(token, max_age=TOKEN_MAX_AGE_SECONDS)
    except SignatureExpired as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API token expired.") from exc
    except BadSignature as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token.") from exc

    role = str(payload.get("role", "")).strip().lower()
    subject = str(payload.get("subject", "")).strip()
    if role not in {"patient", "doctor", "nurse"} or not subject:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token payload.")
    return APIPrincipal(role=role, subject=subject)


def get_api_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> APIPrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API bearer token.")
    return _decode_api_token(credentials.credentials)


def require_api_roles(*roles: str) -> Callable[[APIPrincipal], APIPrincipal]:
    allowed_roles = {str(role).strip().lower() for role in roles if str(role).strip()}

    def dependency(principal: APIPrincipal = Depends(get_api_principal)) -> APIPrincipal:
        if allowed_roles and principal.role not in allowed_roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this API role.")
        return principal

    return dependency


def assert_patient_scope(principal: APIPrincipal, patient_id: str) -> None:
    normalized = str(patient_id).strip()
    if principal.role == "patient" and principal.subject != normalized:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this patient account.")


def assert_doctor_scope(principal: APIPrincipal, doctor_id: str) -> None:
    normalized = str(doctor_id).strip()
    if principal.role == "doctor" and principal.subject != normalized:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this doctor account.")
