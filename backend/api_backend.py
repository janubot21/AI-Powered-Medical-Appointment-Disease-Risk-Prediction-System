import hashlib
import sys
import csv
import hmac
import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # Allow `python backend/api_backend.py` to work by ensuring project root is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.risk_engine import engine
from backend.config import get_settings
from backend import db as dbmod


BASE_DIR = Path(__file__).resolve().parent  # backend/
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
PAGES_DIR = FRONTEND_DIR / "pages"
DATA_DIR = PROJECT_ROOT / "data"
APP_DATA_DIR = DATA_DIR / "app"
DATASETS_DIR = DATA_DIR / "datasets"
FAVICON_ICO_PATH = STATIC_DIR / "favicon.ico"
FAVICON_SVG_PATH = STATIC_DIR / "img" / "heartbeat.svg"
USERS_FILE_CSV = APP_DATA_DIR / "users.csv"
USERS_PATIENT_CSV = APP_DATA_DIR / "users_patient.csv"
USERS_NURSE_CSV = APP_DATA_DIR / "users_nurse.csv"
USERS_DOCTOR_CSV = APP_DATA_DIR / "users_doctor.csv"
USERS_FILE_JSON = APP_DATA_DIR / "users.json"
APPOINTMENTS_FILE = APP_DATA_DIR / "appointments.json"
HEALTH_DETAILS_SUBMISSIONS_FILE = APP_DATA_DIR / "health_details_submissions.csv"
PATIENT_RECORD_HISTORY_FILE = APP_DATA_DIR / "patient_record_history.json"
DOCTOR_LEAVES_FILE = APP_DATA_DIR / "doctor_leaves.csv"
DOCTOR_LEAVE_DATASET_FILE = DATASETS_DIR / "doctor_leave.csv"
DOCTORS_DATASET_FILE = DATASETS_DIR / "doctors.csv"

FRONTEND_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)

DATA_LOCK = threading.RLock()
SETTINGS = get_settings()
DB_CONN = None
_RATE: dict[str, list[float]] = {}


def rate_limit(request: Request, scope: str) -> None:
    try:
        ip = (request.client.host if request.client else "unknown") or "unknown"
    except Exception:
        ip = "unknown"
    key = f"{scope}:{ip}"
    now = datetime.now(timezone.utc).timestamp()
    window = float(SETTINGS.rate_limit_window_s)
    max_hits = int(SETTINGS.rate_limit_max)
    hits = _RATE.get(key, [])
    hits = [t for t in hits if (now - t) <= window]
    hits.append(now)
    _RATE[key] = hits
    if len(hits) > max_hits:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")


def get_db() -> Any:
    global DB_CONN
    if DB_CONN is None:
        DB_CONN = dbmod.connect(SETTINGS.db_path)
        dbmod.ensure_schema(DB_CONN)
        try:
            if not dbmod.list_users(DB_CONN):
                legacy_users: list[dict[str, Any]] = []
                for p in (USERS_PATIENT_CSV, USERS_NURSE_CSV, USERS_DOCTOR_CSV, USERS_FILE_CSV):
                    if p.exists():
                        legacy_users.extend(read_users_csv(p))
                for u in legacy_users:
                    email = normalize_email(u.get("email") or "")
                    if not email or dbmod.get_user_by_email(DB_CONN, email):
                        continue
                    role = safe_normalize_role(u.get("role") or "patient")
                    phone = (u.get("phone") or "").strip()
                    salt = (u.get("salt") or "").strip() or secrets.token_hex(16)
                    password_hash = (u.get("password_hash") or "").strip()
                    if not password_hash:
                        continue
                    created = dbmod.create_user(
                        DB_CONN,
                        full_name=(u.get("full_name") or email).strip(),
                        email=email,
                        phone=phone,
                        role=role,
                        salt=salt,
                        password_hash=password_hash,
                    )
                    details = u.get("health_details")
                    if isinstance(details, dict) and details:
                        dbmod.upsert_health_details(DB_CONN, created["email"], details)

                if APPOINTMENTS_FILE.exists():
                    for appt in read_json_array(APPOINTMENTS_FILE):
                        if not appt or not appt.get("patient_email"):
                            continue
                        try:
                            dbmod.insert_appointment(DB_CONN, appt)
                        except Exception:
                            continue

                if PATIENT_RECORD_HISTORY_FILE.exists():
                    for ev in read_json_array(PATIENT_RECORD_HISTORY_FILE):
                        if not ev or not ev.get("patient_email"):
                            continue
                        try:
                            dbmod.insert_history_event(DB_CONN, ev)
                        except Exception:
                            continue
        except Exception:
            pass
    return DB_CONN

HTML_PAGES: dict[str, str] = {
    "root": "frontend/pages/role-selection.html",
    "patient": "frontend/pages/patient-dashboard.html",
    "patient_appointments": "frontend/pages/patient-appointments.html",
    "patient_book": "frontend/pages/appointment-book.html",
    "patient_history": "frontend/pages/patient-history.html",
    "nurse": "frontend/pages/nurse.html",
    "nurse_editor": "frontend/pages/nurse-editor.html",
    "doctor": "frontend/pages/doctor.html",
    "doctor_past": "frontend/pages/doctor-past.html",
    "doctor_patients": "frontend/pages/doctor-patients.html",
    "doctor_leave": "frontend/pages/doctor-leave.html",
    "login": "frontend/pages/login.html",
    "register": "frontend/pages/register.html",
    "health_details": "frontend/pages/health-details.html",
}


app = FastAPI(
    title="Healthcare Risk Prediction API",
    description="Live appointment-time risk estimation service with Logistic Regression plus real-world clinical threshold checks.",
    version="2.1.0",
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SETTINGS.session_secret,
    same_site="lax",
    https_only=SETTINGS.https_only_cookies,
    session_cookie="app_session",
    max_age=60 * 60 * 24 * 7,
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    if FAVICON_ICO_PATH.exists():
        return FileResponse(path=str(FAVICON_ICO_PATH))

    if FAVICON_SVG_PATH.exists():
        return RedirectResponse(url="/static/img/heartbeat.svg")

    return Response(status_code=204)

ALLOWED_ROLES: tuple[str, ...] = ("patient", "nurse", "doctor")


def normalize_role(role: str) -> str:
    normalized_role = role.strip().lower()
    if normalized_role not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail="Invalid role selected.")
    return normalized_role


def safe_normalize_role(role: str) -> str:
    try:
        return normalize_role(role)
    except HTTPException:
        return "patient"


def dashboard_path_for_role(role: str, has_health_details: bool = False) -> str:
    normalized_role = normalize_role(role)
    if normalized_role == "patient" and not has_health_details:
        return "/health-details"
    return f"/{normalized_role}"


def normalize_email(email: str) -> str:
    return email.strip().lower()


PHONE_CLEAN_RE = re.compile(r"[^\d+]")
PHONE_VALID_RE = re.compile(r"^\+?\d{7,15}$")
EMAIL_VALID_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_phone(phone: str) -> str:
    raw = (phone or "").strip()
    cleaned = PHONE_CLEAN_RE.sub("", raw)
    if cleaned.startswith("++"):
        cleaned = cleaned.lstrip("+")
        cleaned = f"+{cleaned}"
    if cleaned.startswith("+"):
        cleaned = "+" + re.sub(r"\D", "", cleaned[1:])
    else:
        cleaned = re.sub(r"\D", "", cleaned)
    if not cleaned or not PHONE_VALID_RE.match(cleaned):
        raise HTTPException(status_code=400, detail="Invalid phone number format.")
    return cleaned


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized or len(normalized) > 254 or not EMAIL_VALID_RE.match(normalized):
        raise HTTPException(status_code=400, detail="Invalid email format.")
    return normalized


def validate_password_strength(password: str) -> None:
    pw = str(password or "")
    if len(pw) < 8 or len(pw) > 128:
        raise HTTPException(status_code=400, detail="Password must be 8-128 characters long.")
    if any(ch.isspace() for ch in pw):
        raise HTTPException(status_code=400, detail="Password must not contain spaces.")
    if not re.search(r"[a-z]", pw):
        raise HTTPException(status_code=400, detail="Password must include at least 1 lowercase letter.")
    if not re.search(r"[A-Z]", pw):
        raise HTTPException(status_code=400, detail="Password must include at least 1 uppercase letter.")
    if not re.search(r"\d", pw):
        raise HTTPException(status_code=400, detail="Password must include at least 1 number.")
    if not re.search(r"[^A-Za-z0-9]", pw):
        raise HTTPException(status_code=400, detail="Password must include at least 1 special character.")


def get_session_user(request: Request) -> dict[str, Any] | None:
    user = request.session.get("user") if hasattr(request, "session") else None
    return user if isinstance(user, dict) and user.get("email") and user.get("role") else None


def require_session(request: Request) -> dict[str, Any]:
    user = get_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    if request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"}:
        expected = str(request.session.get("csrf") or "").strip()
        if not expected:
            raise HTTPException(status_code=403, detail="CSRF token missing.")
        received = str(request.headers.get(SETTINGS.csrf_header) or "").strip()
        if not received or received != expected:
            raise HTTPException(status_code=403, detail="Invalid CSRF token.")
    return user


def require_role(*roles: str):
    role_set = {str(r).strip().lower() for r in roles if str(r).strip()}

    def dep(user: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
        current = str(user.get("role") or "").strip().lower()
        if role_set and current not in role_set:
            raise HTTPException(status_code=403, detail="Forbidden.")
        return user

    return dep


def html_page(filename: str) -> FileResponse | HTMLResponse:
    path = PROJECT_ROOT / filename
    if path.exists():
        return FileResponse(path, headers=dict(NO_STORE_HEADERS))
    return HTMLResponse(
        content=(
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Missing page</title></head><body style='font-family:system-ui'>"
            "<h2>Frontend page missing</h2>"
            f"<p>Expected file: <code>{filename}</code></p>"
            "<p>Try: <a href='/docs'>/docs</a> or <a href='/health'>/health</a></p>"
            "</body></html>"
        ),
        headers=dict(NO_STORE_HEADERS),
        status_code=404,
    )


def read_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"{path.name} must contain a JSON array.")
    return parsed


def write_json_array(path: Path, payload: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)


TRACKED_HEALTH_DETAIL_FIELDS: tuple[str, ...] = (
    "age",
    "gender",
    "symptoms",
    "symptom_count",
    "height_cm",
    "weight_kg",
    "bmi",
    "smoking_habit",
    "alcohol_habit",
    "average_sleep_hours",
    "medical_history",
    "family_history_major_chronic_disease",
    "glucose",
    "glucose_type",
    "blood_pressure",
)


def normalize_audit_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (int, float)):
        if isinstance(value, bool):
            return value
        try:
            return float(value)
        except Exception:
            return value
    return value


def snapshot_health_details(details: dict[str, Any] | None) -> dict[str, Any]:
    src = details if isinstance(details, dict) else {}
    return {key: src.get(key) for key in TRACKED_HEALTH_DETAIL_FIELDS if src.get(key) is not None and src.get(key) != ""}


def compute_health_details_changes(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    b = before if isinstance(before, dict) else {}
    a = after if isinstance(after, dict) else {}
    changes: list[dict[str, Any]] = []
    for key in TRACKED_HEALTH_DETAIL_FIELDS:
        old_val = normalize_audit_value(b.get(key))
        new_val = normalize_audit_value(a.get(key))
        if old_val != new_val:
            changes.append({"field": key, "from": old_val, "to": new_val})
    return changes


def load_patient_record_history() -> list[dict[str, Any]]:
    # Kept for backward compatibility in call sites that still expect a list.
    # Prefer querying per patient via the DB layer.
    return []


def append_patient_record_history(event: dict[str, Any]) -> None:
    dbmod.insert_history_event(get_db(), event)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader if row]


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: (row.get(name, "") if row.get(name, "") is not None else "") for name in fieldnames})
    tmp_path.replace(path)


DOCTOR_LEAVE_FIELDS = ["id", "doctor_email", "start_at", "end_at", "reason", "created_at", "updated_at"]


def load_doctor_leaves() -> list[dict[str, Any]]:
    return dbmod.list_all_doctor_leaves(get_db())


def save_doctor_leaves(leaves: list[dict[str, Any]]) -> None:
    raise HTTPException(status_code=500, detail="save_doctor_leaves() is deprecated (DB-backed storage).")


def next_leave_id(entries: list[dict[str, Any]]) -> int:
    numeric_ids = [entry.get("id") for entry in entries if isinstance(entry.get("id"), int) and entry.get("id")]
    return (max(numeric_ids) + 1) if numeric_ids else 1


def doctor_leave_for(doctor_email: str, when: datetime) -> dict[str, Any] | None:
    email = normalize_email(doctor_email)
    for leave in dbmod.list_doctor_leaves(get_db(), email):
        try:
            start_at = parse_iso_datetime(leave.get("start_at") or "")
            end_at = parse_iso_datetime(leave.get("end_at") or "")
        except Exception:
            continue
        if start_at <= when <= end_at:
            return leave
    return None


def ensure_dataset_files() -> None:
    if not DOCTORS_DATASET_FILE.exists():
        DOCTORS_DATASET_FILE.write_text(
            "doctor_id,doctor_name,specialization,available_time,emergency_doctor\n",
            encoding="utf-8",
        )


def load_doctors_dataset() -> list[dict[str, Any]]:
    ensure_dataset_files()
    doctors: list[dict[str, Any]] = []
    with DOCTORS_DATASET_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            doctor_id = (row.get("doctor_id") or "").strip()
            if not doctor_id:
                continue
            emergency_raw = (row.get("emergency_doctor") or "").strip().lower()
            doctors.append(
                {
                    "doctor_id": doctor_id,
                    "doctor_name": (row.get("doctor_name") or doctor_id).strip() or doctor_id,
                    "specialization": (row.get("specialization") or "").strip(),
                    "available_time": (row.get("available_time") or "").strip(),
                    "emergency_doctor": emergency_raw in {"yes", "true", "1", "y"},
                }
            )
    return doctors

def doctor_is_on_leave(doctor_id: str, when: datetime) -> bool:
    email = normalize_email(doctor_id)
    if not email:
        return False
    leave = doctor_leave_for(email, when)
    return leave is not None


def emergency_doctors_available(when: datetime) -> list[dict[str, Any]]:
    doctors = [d for d in load_doctors_dataset() if d.get("emergency_doctor")]
    available: list[dict[str, Any]] = []
    for d in doctors:
        if not doctor_is_on_leave(d["doctor_id"], when):
            available.append(d)
    return available


def append_health_details_submission(health_details: dict[str, Any]) -> None:
    email = normalize_email(str(health_details.get("email") or ""))
    submitted_at = str(health_details.get("submitted_at") or datetime.now(timezone.utc).isoformat())
    if not email:
        return
    try:
        dbmod.insert_health_details_submission(get_db(), email=email, submitted_at=submitted_at, payload=health_details)
    except Exception:
        # Best-effort; do not block the main flow on audit storage.
        return


_SEEDED = False


def load_users() -> list[dict[str, Any]]:
    global _SEEDED
    conn = get_db()

    # Seed demo accounts once (development only).
    if not _SEEDED and SETTINGS.env not in {"production", "prod"}:
        demo_password = os.environ.get("APP_DEMO_PASSWORD") or "DemoPass1!"
        demo_users = [
            {
                "full_name": "Patient Demo",
                "email": "patient@example.com",
                "phone": "",
                "role": "patient",
            },
            {
                "full_name": "Nurse Demo",
                "email": "nurse@example.com",
                "phone": "",
                "role": "nurse",
            },
            {
                "full_name": "Doctor Demo",
                "email": "doctor@example.com",
                "phone": "",
                "role": "doctor",
            },
        ]
        for demo in demo_users:
            existing = dbmod.get_user_by_email(conn, demo["email"])
            try:
                salt = secrets.token_hex(16)
                password_hash = hash_password(demo_password, salt)
                if existing:
                    dbmod.update_user_password(conn, demo["email"], salt, password_hash)
                else:
                    dbmod.create_user(
                        conn,
                        full_name=demo["full_name"],
                        email=demo["email"],
                        phone=demo["phone"],
                        role=demo["role"],
                        salt=salt,
                        password_hash=password_hash,
                    )
            except Exception:
                pass
        _SEEDED = True

    return dbmod.list_users(conn)


def save_users(users: list[dict[str, Any]]) -> None:
    # Storage has moved to SQLite; this function is kept only to avoid accidental call sites.
    # Any remaining calls should be refactored to use db operations.
    raise HTTPException(status_code=500, detail="save_users() is deprecated (DB-backed storage).")


def load_appointments() -> list[dict[str, Any]]:
    return dbmod.list_all_appointments(get_db())


def save_appointments(appointments: list[dict[str, Any]]) -> None:
    raise HTTPException(status_code=500, detail="save_appointments() is deprecated (DB-backed storage).")


def next_numeric_id(entries: list[dict[str, Any]]) -> int:
    numeric_ids = [entry.get("id") for entry in entries if isinstance(entry.get("id"), int)]
    return (max(numeric_ids) + 1) if numeric_ids else 1


def next_role_user_id(users: list[dict[str, Any]], role: str) -> int:
    normalized_role = safe_normalize_role(role)
    role_users = [user for user in users if safe_normalize_role(user.get("role") or "patient") == normalized_role]
    return next_numeric_id(role_users)


def get_user_by_email(email: str) -> dict[str, Any] | None:
    return dbmod.get_user_by_email(get_db(), normalize_email(email))


def get_user_by_phone(phone: str) -> dict[str, Any] | None:
    return dbmod.get_user_by_phone(get_db(), normalize_phone(phone))


def create_user(full_name: str, email: str, phone: str, role: str, salt: str, password_hash: str) -> None:
    normalized_email = normalize_email(email)
    normalized_phone = normalize_phone(phone)
    normalized_role = normalize_role(role)
    conn = get_db()
    if dbmod.get_user_by_email(conn, normalized_email):
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    if normalized_phone and dbmod.get_user_by_phone(conn, normalized_phone):
        raise HTTPException(status_code=409, detail="An account with this phone number already exists.")
    dbmod.create_user(
        conn,
        full_name=full_name,
        email=normalized_email,
        phone=normalized_phone,
        role=normalized_role,
        salt=salt,
        password_hash=password_hash,
    )


def update_user_health_details(email: str, health_details: dict[str, Any]) -> None:
    normalized_email = normalize_email(email)
    conn = get_db()
    if not dbmod.get_user_by_email(conn, normalized_email):
        raise HTTPException(status_code=404, detail="User account not found.")
    dbmod.upsert_health_details(conn, normalized_email, health_details)


def get_all_users() -> list[dict[str, Any]]:
    return dbmod.list_users(get_db())


def hash_password(password: str, salt: str) -> str:
    iterations = 200_000
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return f"pbkdf2_sha256${iterations}${salt}${digest}"


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    raw = str(stored_hash or "").strip()
    if not raw:
        return False
    if raw.startswith("pbkdf2_sha256$"):
        parts = raw.split("$")
        if len(parts) != 4:
            return False
        try:
            iterations = int(parts[1])
        except Exception:
            return False
        salt_from_hash = parts[2]
        expected = parts[3]
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_from_hash.encode("utf-8"),
            iterations,
        ).hex()
        return hmac.compare_digest(digest, expected)

    # Legacy: raw hex digest with salt stored separately.
    # Prefer PBKDF2 (older iterations), then fall back to SHA-256(salt+password) if needed.
    digest_legacy_pbkdf2 = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        str(salt or "").encode("utf-8"),
        100_000,
    ).hex()
    if hmac.compare_digest(digest_legacy_pbkdf2, raw):
        return True
    digest_sha256 = hashlib.sha256((str(salt or "") + password).encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest_sha256, raw)


def find_user(users: list[dict[str, Any]], email: str) -> dict[str, Any] | None:
    normalized = normalize_email(email)
    return next(
        (entry for entry in users if normalize_email(str(entry.get("email", ""))) == normalized),
        None,
    )


def find_user_by_phone(users: list[dict[str, Any]], phone: str) -> dict[str, Any] | None:
    normalized = normalize_phone(phone)
    for entry in users:
        stored = (entry.get("phone") or "").strip()
        if not stored:
            continue
        try:
            if normalize_phone(stored) == normalized:
                return entry
        except HTTPException:
            # Ignore malformed stored phone values rather than crashing lookups.
            continue
    return None


def read_users_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        users: list[dict[str, Any]] = []
        for row in reader:
            if not row:
                continue
            health_details_raw = (row.get("health_details") or "").strip()
            health_details: dict[str, Any] | None = None
            if health_details_raw:
                try:
                    parsed = json.loads(health_details_raw)
                    if isinstance(parsed, dict):
                        health_details = parsed
                except json.JSONDecodeError:
                    health_details = None

            user_id = row.get("id")
            try:
                parsed_id = int(user_id) if user_id not in (None, "") else None
            except ValueError:
                parsed_id = None

            users.append(
                {
                    "id": parsed_id,
                    "full_name": (row.get("full_name") or "").strip(),
                    "email": normalize_email(row.get("email") or ""),
                    "phone": (row.get("phone") or "").strip(),
                    "role": safe_normalize_role(row.get("role") or "patient")
                    if (row.get("role") or "").strip()
                    else "patient",
                    "salt": (row.get("salt") or "").strip(),
                    "password_hash": (row.get("password_hash") or "").strip(),
                    "health_details": health_details,
                }
            )
        return users


def write_users_csv(path: Path, users: list[dict[str, Any]]) -> None:
    fieldnames = ["id", "full_name", "email", "phone", "role", "salt", "password_hash", "health_details"]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for user in users:
            health_details = user.get("health_details")
            phone_raw = (user.get("phone") or "").strip()
            phone_out = ""
            if phone_raw:
                try:
                    phone_out = normalize_phone(phone_raw)
                except HTTPException:
                    phone_out = phone_raw
            writer.writerow(
                {
                    "id": user.get("id") or "",
                    "full_name": user.get("full_name") or "",
                    "email": normalize_email(user.get("email") or ""),
                    "phone": phone_out,
                    "role": safe_normalize_role(user.get("role") or "patient"),
                    "salt": user.get("salt") or "",
                    "password_hash": user.get("password_hash") or "",
                    "health_details": json.dumps(health_details) if isinstance(health_details, dict) else "",
                }
            )
    tmp_path.replace(path)


def normalize_role_ids(users: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    # Stable ordering: by current id (if any), then email.
    def sort_key(user: dict[str, Any]) -> tuple[int, str]:
        user_id = user.get("id")
        sortable_id = int(user_id) if isinstance(user_id, int) else 1_000_000_000
        return (sortable_id, normalize_email(user.get("email") or ""))

    changed = False
    normalized: list[dict[str, Any]] = []
    for new_id, user in enumerate(sorted(users, key=sort_key), start=1):
        cloned = dict(user)
        if cloned.get("id") != new_id:
            cloned["id"] = new_id
            changed = True
        normalized.append(cloned)
    return normalized, changed


def ensure_seed_users(users: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    seeded = list(users)
    changed = False
    existing_emails = {normalize_email(u.get("email", "")) for u in seeded if u.get("email")}
    demo_users = [
        {
            "id": 1,
            "full_name": "Patient Demo",
            "email": "patient@example.com",
            "role": "patient",
            "salt": "patient_salt_v1",
            "password_hash": "aebcb5af2f38c215c62864d0a278ab908a0fb4e8c4ce23422e74a9e5e699b3ec",
            "health_details": None,
        },
        {
            "id": 2,
            "full_name": "Nurse Demo",
            "email": "nurse@example.com",
            "role": "nurse",
            "salt": "nurse_salt_v1",
            "password_hash": "5c608afadf2433b1baca47bd9d2ec6f23d517440ec8776f7b71950352b710ba2",
            "health_details": None,
        },
        {
            "id": 3,
            "full_name": "Doctor Demo",
            "email": "doctor@example.com",
            "role": "doctor",
            "salt": "doctor_salt_v1",
            "password_hash": "e52eba4abd441d394568acced08dc513155fe97e42711bc16d44f6f61ebee28e",
            "health_details": None,
        },
    ]

    for demo in demo_users:
        demo_email = normalize_email(demo["email"])
        if demo_email in existing_emails:
            continue
        seeded.append(demo)
        existing_emails.add(demo_email)
        changed = True

    if changed:
        # Persist happens via save_users() in load_users.
        return seeded, True
    return seeded, False


def user_public_payload(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": user.get("id"),
        "full_name": user["full_name"],
        "email": user["email"],
        "phone": user.get("phone") or "",
        "role": user.get("role", "patient"),
        "dashboard_path": dashboard_path_for_role(
            user.get("role", "patient"),
            has_health_details=bool(user.get("health_details")),
        ),
        "health_details_completed": bool(user.get("health_details")),
    }


def user_directory_payload(user: dict[str, Any]) -> dict[str, Any]:
    return {
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user.get("role", "patient"),
        "health_details_completed": bool(user.get("health_details")),
        "health_details": user.get("health_details"),
    }


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
    phone: str = Field(..., min_length=7, max_length=20, description="User phone number.")
    password: str = Field(..., min_length=8, max_length=128, description="Account password.")
    role: Literal["patient", "nurse", "doctor"] = Field(..., description="Account role.")


class LoginRequest(BaseModel):
    email: str | None = Field(None, min_length=5, max_length=254, description="User email address.")
    phone: str | None = Field(None, min_length=7, max_length=20, description="User phone number.")
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

def parse_iso_datetime(value: str) -> datetime:
    # Accept "Z" suffix and naive strings; store normalized UTC.
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def appointment_to_payload(appointment: dict[str, Any]) -> dict[str, Any]:
    patient = get_user_by_email(appointment["patient_email"])
    risk = compute_patient_risk(patient.get("health_details") if patient else None)
    return {
        "id": int(appointment["id"]),
        "patient_email": appointment["patient_email"],
        "patient_name": patient["full_name"] if patient else appointment["patient_email"],
        "scheduled_for": appointment["scheduled_for"],
        "status": appointment.get("status", "Pending"),
        "appointment_type": appointment.get("appointment_type"),
        "doctor_email": appointment.get("doctor_email"),
        "contact_info": appointment.get("contact_info"),
        "priority": appointment.get("priority"),
        "predicted_risk_level": risk.get("risk_level"),
        "predicted_risk_source": risk.get("source"),
        "booking_priority": risk.get("booking_priority"),
        "health_details_submitted_at": (patient.get("health_details") or {}).get("submitted_at") if patient else None,
        "reason": appointment.get("reason") or appointment.get("appointment_type") or "",
        "notes": appointment.get("notes"),
        "created_at": appointment.get("created_at") or "",
    }


def parse_blood_pressure(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) < 2:
        return None

    try:
        systolic = float(numbers[0])
        diastolic = float(numbers[1])
    except ValueError:
        return None

    if systolic <= 0 or diastolic <= 0:
        return None
    return (systolic, diastolic)


def compute_patient_risk(health_details: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(health_details, dict) or not health_details:
        return {}

    age = health_details.get("age")
    gender = health_details.get("gender") or "Other"
    symptoms = health_details.get("symptoms") or ""
    bmi = health_details.get("bmi")
    glucose = health_details.get("glucose")
    bp = parse_blood_pressure(health_details.get("blood_pressure"))

    quick: dict[str, Any] | None = None
    try:
        if isinstance(age, int) and symptoms:
            quick = quick_risk_from_age_symptoms(age=age, symptoms=str(symptoms))
    except Exception:
        quick = None

    # Prefer the full model only when we have the required vitals.
    if (
        isinstance(age, int)
        and isinstance(bmi, (int, float))
        and isinstance(glucose, (int, float))
        and bp is not None
        and symptoms
    ):
        try:
            systolic_bp, diastolic_bp = bp
            model_result = engine.predict(
                age=age,
                gender=str(gender),
                symptoms=str(symptoms),
                glucose=float(glucose),
                glucose_type="fasting",
                systolic_bp=float(systolic_bp),
                diastolic_bp=float(diastolic_bp),
                bmi=float(bmi),
            )

            group = (model_result.get("risk_group") or "").strip()
            # Normalize to the 3-level UI.
            normalized = "Low"
            if group in ("Critical", "High"):
                normalized = "High"
            elif group == "Moderate":
                normalized = "Medium"
            elif group == "Low":
                normalized = "Low"
            else:
                normalized = quick.get("risk_level") if quick else "Low"

            # Symptom-only high-risk keyword override: if the quick symptom scan is High,
            # prefer High even when the full model predicts lower.
            booking_priority = model_result.get("booking_priority")
            if quick and str(quick.get("risk_level") or "").strip() == "High" and normalized != "High":
                normalized = "High"
                booking_priority = "Urgent review recommended"

            return {
                "risk_level": normalized,
                "source": "model",
                "booking_priority": booking_priority,
            }
        except Exception:
            # Fall back to quick estimate.
            pass

    if quick:
        risk_level = str(quick.get("risk_level") or "Low")
        if risk_level == "High":
            booking_priority = "Urgent review recommended"
        elif risk_level == "Medium":
            booking_priority = "Priority follow-up recommended"
        else:
            booking_priority = "Standard scheduling is acceptable"
        return {"risk_level": risk_level, "source": "quick", "booking_priority": booking_priority}

    return {}


def normalize_symptom_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        text = ", ".join([str(v) for v in value if v is not None])
    else:
        text = str(value)
    parts = [re.sub(r"\s+", " ", part).strip().lower() for part in re.split(r"[,;|\n]+", text)]
    return [part for part in parts if part]


def nurse_rule_assessment_from_inputs(
    *,
    age: int,
    gender: str,
    symptoms: str | list[str],
    glucose: float | None,
    glucose_type: str | None,
    systolic_bp: float | None,
    diastolic_bp: float | None,
    bmi: float | None,
) -> dict[str, Any]:
    validation_errors: list[str] = []

    # Nurse-facing validation (prevents obviously unsafe / impossible entries).
    if age <= 0 or age > 120:
        validation_errors.append("Invalid age (must be 1–120).")

    if systolic_bp is None or diastolic_bp is None:
        validation_errors.append("Blood pressure is missing.")
    else:
        if systolic_bp < 50 or systolic_bp > 250:
            validation_errors.append("Invalid systolic BP (must be 50–250).")
        if diastolic_bp < 30 or diastolic_bp > 150:
            validation_errors.append("Invalid diastolic BP (must be 30–150).")

    if glucose is None:
        validation_errors.append("Glucose is missing.")
    else:
        if glucose <= 0:
            validation_errors.append("Invalid glucose (must be > 0).")
        if glucose > 1200:
            validation_errors.append("Invalid glucose (value too high).")

    if bmi is None:
        validation_errors.append("BMI is missing.")
    else:
        if bmi <= 10 or bmi > 60:
            validation_errors.append("Invalid BMI (must be 10–60).")

    symptom_tokens = normalize_symptom_tokens(symptoms)
    symptom_count = len(symptom_tokens)

    stop_words = {"in", "on", "of", "the", "a", "an", "and", "with", "to", "for", "at"}

    def symptom_has(phrase: str) -> bool:
        target = str(phrase or "").strip().lower()
        if not target:
            return False

        target_words = [w for w in re.findall(r"[a-z0-9]+", target) if w and w not in stop_words]
        for token in symptom_tokens:
            if target in token:
                return True
            token_words = [w for w in re.findall(r"[a-z0-9]+", token) if w and w not in stop_words]
            token_set = set(token_words)
            if target_words and all(w in token_set for w in target_words):
                return True
        return False

    symptom_quick: dict[str, Any] | None = None
    try:
        symptom_quick = quick_risk_from_age_symptoms(age=age, symptoms=str(symptoms))
    except Exception:
        symptom_quick = None
    symptom_keyword_high = bool(symptom_quick and str(symptom_quick.get("risk_level") or "").strip() == "High")

    emergency_alerts: list[str] = []
    # Symptom-only emergency triggers (still valid even before vitals are recorded).
    if symptom_has("unconscious") or symptom_has("unconsciousness"):
        emergency_alerts.append("Critical condition: unconscious.")
    if symptom_has("chest pain") and symptom_has("shortness of breath"):
        emergency_alerts.append("Possible heart attack: chest pain + shortness of breath.")
    if symptom_count >= 5:
        emergency_alerts.append("Severe symptom load (5+ symptoms).")

    tags: list[str] = []
    if symptom_has("fever") and symptom_has("cough"):
        tags.append("Infection suspected")

    # If vitals are invalid/missing, return a minimal rule response without engine feature mapping.
    if validation_errors:
        triage_level = "Critical" if emergency_alerts else ("High" if symptom_keyword_high else "Needs vitals")
        show_on_top = triage_level in {"Critical", "High"}
        if triage_level == "Critical":
            appointment_recommendation = "Immediate (0–10 mins)"
        elif triage_level == "High":
            appointment_recommendation = "Within 1 hour (vitals ASAP)"
        else:
            appointment_recommendation = "Vitals required"
        return {
            "valid": False,
            "validation_errors": validation_errors,
            "triage_level": triage_level,
            "priority": "Immediate attention"
            if triage_level == "Critical"
            else ("Urgent consultation (symptoms)" if triage_level == "High" else "Vitals required"),
            "show_on_top": show_on_top,
            "emergency_alerts": emergency_alerts,
            "tags": tags,
            "appointment_recommendation": appointment_recommendation,
            "suggestions": [
                "Confirm vitals using correct equipment and technique.",
                "If the patient looks unwell or symptoms are worrying, escalate per clinic protocol.",
            ],
            "nurse_comment": (
                "Escalate immediately and follow emergency protocol."
                if triage_level == "Critical"
                else (
                    "High-risk symptoms present. Capture vitals urgently and fast-track for review."
                    if triage_level == "High"
                    else "Complete vitals entry to enable full triage."
                )
            ),
            "derived": {
                "symptom_count": symptom_count,
            },
        }

    # Full triage using rule-based clinical thresholds already implemented in the engine.
    features, clinical = engine.build_feature_row(
        age=age,
        gender=gender,
        symptoms=symptoms,
        glucose=float(glucose),
        glucose_type=str(glucose_type or "fasting"),
        systolic_bp=float(systolic_bp),
        diastolic_bp=float(diastolic_bp),
        bmi=float(bmi),
    )
    clinical_bp_category = str(clinical["blood_pressure"]["clinical_category"])
    clinical_glucose_category = str(clinical["glucose"]["clinical_category"])
    clinical_bmi_category = str(clinical["bmi"]["clinical_category"])
    clinical_risk_group = str(clinical["clinical_risk_group"])

    # Emergency / critical conditions (top priority)
    if float(systolic_bp) > 180 or float(diastolic_bp) > 120:
        emergency_alerts.append("Hypertensive crisis (BP > 180/120).")
    if float(glucose) < 54:
        emergency_alerts.append("Severe hypoglycemia (glucose < 54 mg/dL).")
    if float(glucose) > 300:
        emergency_alerts.append("Severe hyperglycemia (glucose > 300 mg/dL).")

    triage_level = "Critical" if emergency_alerts else clinical_risk_group

    # High risk priority (fast track)
    if triage_level != "Critical":
        if clinical_bp_category in {"Stage 2 Hypertension", "Severe Hypertension"}:
            triage_level = "High"
        if clinical_glucose_category == "Diabetes Range":
            triage_level = "High"
        if symptom_keyword_high:
            triage_level = "High"

    # Symptom-based tags
    if symptom_has("headache") and clinical_bp_category in {
        "Stage 1 Hypertension",
        "Stage 2 Hypertension",
        "Severe Hypertension",
    }:
        tags.append("Hypertension-related")
    if symptom_has("fatigue") and clinical_glucose_category == "Diabetes Range":
        tags.append("Diabetes-related")
    if symptom_count >= 4:
        tags.append("Multi-system issue")

    # Appointment scheduling rules
    if triage_level == "Critical":
        appointment_recommendation = "Immediate (0–10 mins)"
        priority = "Immediate emergency review recommended"
    elif triage_level == "High":
        appointment_recommendation = "Within 1 hour"
        priority = "Urgent consultation"
    elif triage_level == "Moderate":
        appointment_recommendation = "Same day"
        priority = "Priority follow-up"
    else:
        appointment_recommendation = "Normal booking"
        priority = "Normal booking"

    feature_row = features.iloc[0].to_dict()
    repeated_symptom_pattern = int(feature_row.get("Repeated_Symptom_Pattern") or 0)
    historical_pattern_score = int(feature_row.get("Historical_Pattern_Score") or 0)

    notes: list[str] = []
    if repeated_symptom_pattern == 1:
        notes.append("Recurring symptom pattern seen in the reference dataset (pattern match).")
    if historical_pattern_score >= 2:
        notes.append("Reference patterns suggest a possible chronic trend (not a diagnosis).")

    suggestions: list[str] = []
    if float(glucose) < 70:
        suggestions.append("If confirmed low glucose, provide fast-acting carbohydrate per protocol and recheck.")
    if symptom_has("fever"):
        suggestions.append("Supportive care and temperature monitoring per protocol.")
    if clinical_bp_category in {"Stage 1 Hypertension", "Stage 2 Hypertension", "Severe Hypertension"}:
        suggestions.append("Recheck BP after rest using correct cuff size and positioning.")

    if triage_level == "Critical":
        nurse_comment = "Escalate immediately and follow emergency protocol."
    elif triage_level == "High":
        nurse_comment = "Fast-track for urgent evaluation and document vitals + key symptoms."
    elif triage_level == "Moderate":
        nurse_comment = "Schedule same-day review; monitor symptoms and recheck vitals if needed."
    else:
        nurse_comment = "Proceed with normal workflow; provide standard precautions and follow-up guidance."

    show_on_top = bool(emergency_alerts) or bool(clinical.get("urgent_flag"))

    return {
        "valid": True,
        "validation_errors": [],
        "triage_level": triage_level,
        "priority": priority,
        "show_on_top": show_on_top,
        "emergency_alerts": emergency_alerts,
        "tags": sorted(set(tags)),
        "appointment_recommendation": appointment_recommendation,
        "suggestions": suggestions,
        "nurse_comment": nurse_comment,
        "derived": {
            "symptom_count": int(clinical["symptoms"]["symptom_count"]),
            "blood_pressure_category": clinical_bp_category,
            "glucose_category": clinical_glucose_category,
            "bmi_category": clinical_bmi_category,
            "clinical_risk_group": clinical_risk_group,
            "repeated_symptom_pattern": repeated_symptom_pattern,
            "historical_pattern_score": historical_pattern_score,
            "notes": notes,
        },
    }


def nurse_rule_assessment_from_health_details(health_details: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(health_details, dict) or not health_details:
        return None

    bp = parse_blood_pressure(health_details.get("blood_pressure"))
    systolic_bp = bp[0] if bp else None
    diastolic_bp = bp[1] if bp else None

    return nurse_rule_assessment_from_inputs(
        age=int(health_details.get("age") or 0),
        gender=str(health_details.get("gender") or "Other"),
        symptoms=str(health_details.get("symptoms") or ""),
        glucose=(float(health_details.get("glucose")) if health_details.get("glucose") is not None else None),
        glucose_type=str(health_details.get("glucose_type") or "fasting"),
        systolic_bp=(float(systolic_bp) if systolic_bp is not None else None),
        diastolic_bp=(float(diastolic_bp) if diastolic_bp is not None else None),
        bmi=(float(health_details.get("bmi")) if health_details.get("bmi") is not None else None),
    )

def split_symptoms_text(symptoms: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;|\n]+", symptoms or "") if part.strip()]


def quick_risk_from_age_symptoms(age: int, symptoms: str) -> dict[str, Any]:
    # NOTE: By user request, this quick risk is symptom-based only.
    symptom_list = split_symptoms_text(symptoms or "")
    normalized = [s.lower().strip() for s in symptom_list if s.strip()]

    stop_words = {"in", "on", "of", "the", "a", "an", "and", "with", "to", "for", "at"}

    def matches(symptom_text: str, keyword: str) -> bool:
        text = str(symptom_text or "").strip().lower()
        target = str(keyword or "").strip().lower()
        if not text or not target:
            return False

        # Fast path (also catches variants like "feverish" for keyword "fever").
        if target in text:
            return True

        text_words = [w for w in re.findall(r"[a-z0-9]+", text) if w and w not in stop_words]
        target_words = [w for w in re.findall(r"[a-z0-9]+", target) if w and w not in stop_words]
        if not target_words:
            return False
        text_set = set(text_words)
        return all(w in text_set for w in target_words)

    high_keywords = [
        "chest pain",
        "difficulty breathing",
        "unconsciousness",
        "unconscious",
        "severe bleeding",
        "seizure",
        "seizures",
        "slurred speech",
        "confusion",
        "bluish lips",
        "bluish face",
        "blue lips",
        "blue face",
        "severe head injury",
        "head injury",
        "uncontrolled vomiting",
        "sudden weakness",
        "one side weakness",
        "weakness one side",
        "stroke",
    ]

    medium_keywords = [
        "moderate fever",
        "persistent cough",
        "shortness of breath",
        "dehydration",
        "abdominal pain",
        "dizziness",
        "extreme fatigue",
        "fatigue (extreme)",
        "swelling",
        "rash with fever",
        "frequent vomiting",
        "painful urination",
    ]

    low_keywords = [
        "mild fever",
        "runny nose",
        "sore throat",
        "mild cough",
        "headache",
        "body aches",
        "mild fatigue",
        "sneezing",
        "slight nausea",
        "minor cuts",
        "bruises",
        "minor cuts/bruises",
    ]

    def has_any(substr: str) -> bool:
        return any(matches(s, substr) for s in normalized)

    # Special handling for fever/cough/vomiting intensity words.
    has_fever = has_any("fever")
    fever_is_high = has_any("high fever") or has_any("very high") or has_any("persistent fever")
    fever_is_moderate = has_any("moderate fever")
    fever_is_mild = has_any("mild fever")

    cough_is_persistent = has_any("persistent cough")
    cough_is_mild = has_any("mild cough")

    vomiting_is_uncontrolled = has_any("uncontrolled vomiting")
    vomiting_is_frequent = has_any("frequent vomiting")

    # Compute matches.
    matched_high: list[str] = []
    matched_medium: list[str] = []
    matched_low: list[str] = []

    for kw in high_keywords:
        if kw and has_any(kw):
            matched_high.append(kw)

    for kw in medium_keywords:
        if kw and has_any(kw):
            matched_medium.append(kw)

    for kw in low_keywords:
        if kw and has_any(kw):
            matched_low.append(kw)

    # Map intensity-derived symptoms into buckets.
    if has_fever:
        if fever_is_high:
            matched_high.append("high fever")
        elif fever_is_moderate:
            matched_medium.append("moderate fever")
        elif fever_is_mild:
            matched_low.append("mild fever")
        else:
            # Default fever severity if no qualifier provided.
            matched_medium.append("fever")

    if cough_is_persistent:
        matched_medium.append("persistent cough")
    elif cough_is_mild:
        matched_low.append("mild cough")

    if vomiting_is_uncontrolled:
        matched_high.append("uncontrolled vomiting")
    elif vomiting_is_frequent or has_any("vomiting"):
        matched_medium.append("frequent vomiting")

    # Combination rule requested:
    # - high + low => high
    # - medium + low => medium
    # - any high => high
    # - else any medium => medium
    # - else any low => low
    severity = "Low"
    if matched_high:
        severity = "High"
    elif matched_medium:
        severity = "Medium"
    elif matched_low:
        severity = "Low"
    else:
        severity = "Low"

    # Score is only for display; severity is the decision.
    score = 0
    score += 3 * len(set(matched_high))
    score += 1 * len(set(matched_medium))
    score += 0 * len(set(matched_low))

    triggers: list[str] = []
    if matched_high:
        triggers.append("Matched emergency warning sign symptom(s).")
    elif matched_medium:
        triggers.append("Matched symptoms needing medical attention soon.")
    elif matched_low:
        triggers.append("Matched mild/self-care symptoms.")

    # Include the exact matched keywords (deduped).
    seen_kw: set[str] = set()
    for kw in matched_high + matched_medium + matched_low:
        if kw in seen_kw:
            continue
        seen_kw.add(kw)
        triggers.append(kw)

    if severity == "High":
        advice = "High risk: consider prompt medical evaluation, especially if symptoms are severe or worsening."
    elif severity == "Medium":
        advice = "Medium risk: book a medical review soon and monitor symptoms."
    else:
        advice = "Low risk: try basic self-care and monitor. Book a visit if symptoms persist or worsen."

    return {
        "risk_level": severity,
        "score": score,
        "symptom_count": len(symptom_list),
        "symptoms": ", ".join(symptom_list),
        "triggers": triggers[:10],
        "advice": advice,
        "note": "Quick estimate based only on symptoms (not a medical diagnosis).",
        "matched": {
            "high": sorted(set(matched_high)),
            "medium": sorted(set(matched_medium)),
            "low": sorted(set(matched_low)),
        },
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "healthcare-risk-prediction-api",
        "model": "trained_risk_stratification_model.pkl",
        "rules": "adult clinical glucose, blood pressure, and BMI thresholds enabled",
    }


@app.get("/")
def root() -> FileResponse:
    return html_page(HTML_PAGES["root"])


@app.get("/patient")
def patient_dashboard() -> FileResponse:
    return html_page(HTML_PAGES["patient"])


@app.get("/patient/risk-analysis")
def patient_risk_analysis_dashboard() -> FileResponse:
    # Uses the same UI as /patient, but provides a dedicated URL for the risk view.
    return html_page(HTML_PAGES["patient"])


@app.get("/patient/history")
def patient_history_dashboard() -> FileResponse:
    return html_page(HTML_PAGES["patient_history"])


@app.get("/patient/appointments")
def patient_appointments_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "patient":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["patient_appointments"])


@app.get("/patient/book")
def patient_book_appointment(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "patient":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["patient_book"])


@app.get("/patient/risk")
def patient_risk_dashboard() -> RedirectResponse:
    return RedirectResponse(url="/patient/risk-analysis")


@app.get("/nurse")
def nurse_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "nurse":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["nurse"])


@app.get("/nurse/editor")
def nurse_record_editor(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "nurse":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["nurse_editor"])


@app.get("/doctor")
def doctor_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor"])


@app.get("/doctor/past")
def doctor_past_dashboard(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor_past"])


@app.get("/doctor/patients")
def doctor_patients_page(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor_patients"])


@app.get("/doctor/leave")
def doctor_leave_page(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "doctor":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["doctor_leave"])


@app.get("/login")
def login_page() -> FileResponse:
    return html_page(HTML_PAGES["login"])


@app.get("/register")
def register_page() -> FileResponse:
    return html_page(HTML_PAGES["register"])


@app.get("/health-details")
def health_details_page(request: Request) -> Response:
    user = get_session_user(request)
    if not user or str(user.get("role") or "") != "patient":
        return RedirectResponse(url="/login")
    return html_page(HTML_PAGES["health_details"])


@app.get("/predict-risk")
def predict_risk_form() -> RedirectResponse:
    return RedirectResponse(url="/nurse/editor")


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": "Logistic Regression",
        "target": "Risk_Factor_Group",
        "classes": list(engine.model.classes_),
        "final_risk_benchmarks": ["Low", "Moderate", "High", "Critical"],
        "api_inputs": [
            "age",
            "gender",
            "symptoms",
            "glucose",
            "glucose_type",
            "systolic_bp",
            "diastolic_bp",
            "bmi",
        ],
        "notes": [
            "Blood pressure uses adult clinical systolic/diastolic thresholds.",
            "Glucose uses adult fasting or random glucose thresholds.",
            "The historical ML model was trained on legacy tabular features, so the engine maps live clinical inputs into the model's expected feature schema.",
        ],
    }


@app.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(request: Request, payload: RiskPredictionRequest) -> RiskPredictionResponse:
    require_role("patient", "nurse", "doctor")(require_session(request))
    result = engine.predict(
        age=payload.age,
        gender=payload.gender,
        symptoms=payload.symptoms,
        glucose=payload.glucose,
        glucose_type=payload.glucose_type,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        bmi=payload.bmi,
    )

    # Symptom-only high-risk keyword override (e.g., "difficulty in breathing" should still map to High).
    try:
        symptom_text = (
            ", ".join([str(s) for s in payload.symptoms]) if isinstance(payload.symptoms, list) else str(payload.symptoms)
        )
        symptom_quick = quick_risk_from_age_symptoms(age=payload.age, symptoms=symptom_text)
        if str(symptom_quick.get("risk_level") or "").strip() == "High":
            current = str(result.get("risk_group") or "").strip()
            if current in {"Low", "Moderate"}:
                result["risk_group"] = "High"
                result["booking_priority"] = "Urgent review recommended"
    except Exception:
        pass

    result["nurse_rules"] = nurse_rule_assessment_from_inputs(
        age=payload.age,
        gender=payload.gender,
        symptoms=payload.symptoms,
        glucose=payload.glucose,
        glucose_type=payload.glucose_type,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        bmi=payload.bmi,
    )
    return RiskPredictionResponse(**result)


@app.post("/nurse/triage", response_model=NurseRuleAssessment)
def nurse_triage(request: Request, payload: RiskPredictionRequest) -> NurseRuleAssessment:
    require_role("nurse", "doctor")(require_session(request))
    assessment = nurse_rule_assessment_from_inputs(
        age=payload.age,
        gender=payload.gender,
        symptoms=payload.symptoms,
        glucose=payload.glucose,
        glucose_type=payload.glucose_type,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        bmi=payload.bmi,
    )
    return NurseRuleAssessment(**assessment)


@app.get("/risk/quick")
def quick_risk(age: int = Query(..., ge=0, le=120), symptoms: str = Query(default="")) -> dict[str, Any]:
    return quick_risk_from_age_symptoms(age=age, symptoms=symptoms)


@app.post("/auth/register")
def register_user(payload: RegisterRequest, request: Request) -> dict[str, Any]:
    rate_limit(request, "auth_register")
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
        full_name=payload.full_name,
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
            "full_name": payload.full_name.strip(),
            "email": email,
            "phone": phone,
            "role": role,
            "dashboard_path": dashboard_path_for_role(role, has_health_details=False),
            "health_details_completed": False,
        },
    }


@app.post("/auth/login")
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


@app.post("/auth/logout")
def logout_user(request: Request) -> dict[str, Any]:
    require_session(request)
    try:
        request.session.clear()
    except Exception:
        pass
    return {"message": "Logged out."}


@app.post("/auth/password-reset/request")
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


@app.post("/auth/password-reset/confirm")
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


@app.get("/auth/user")
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


@app.get("/auth/me")
def auth_me(request: Request) -> dict[str, Any]:
    user = require_session(request)
    return {"user": user}


@app.get("/auth/health-details")
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


@app.post("/auth/health-details")
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


@app.get("/api/patient/history")
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


@app.post("/api/patient/health-details/update")
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


@app.get("/portal/directory")
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
    seen: set[str] = set()
    for doc in doctors_from_users:
        key = normalize_email(doc.get("email") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        if when is not None:
            leave = doctor_leave_for(key, when)
            doc = {**doc, "on_leave": bool(leave), "leave": leave, "available": not bool(leave)}
        doctors.append(doc)

    # If there are no doctor accounts yet, fall back to datasets/doctors.csv (doctor_id list).
    for d in load_doctors_dataset():
        key = (d.get("doctor_id") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        leave = doctor_leave_for(key, when) if when is not None else None
        doctors.append(
            {
                "full_name": d.get("doctor_name") or key,
                "email": key,
                "role": "doctor",
                "health_details_completed": False,
                "health_details": None,
                "specialization": d.get("specialization") or "",
                "available_time": d.get("available_time") or "",
                "emergency_doctor": bool(d.get("emergency_doctor")),
                "on_leave": bool(leave) if when is not None else False,
                "leave": leave,
                "available": (not bool(leave)) if when is not None else True,
            }
        )

    return {
        "scheduled_for": when.isoformat() if when is not None else None,
        "patients": patients if role in {"nurse", "doctor"} else [],
        "nurses": nurses if role in {"nurse", "doctor"} else [],
        "doctors": doctors,
    }


@app.get("/api/doctor/leave/check")
def doctor_leave_check(
    request: Request,
    doctor_email: str = Query(..., min_length=5, max_length=254),
    scheduled_for: str = Query(..., min_length=10, max_length=64),
) -> dict[str, Any]:
    require_role("patient", "nurse", "doctor")(require_session(request))
    when = parse_iso_datetime(scheduled_for)
    doctor_id = (doctor_email or "").strip()
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


@app.get("/api/doctor/leaves")
def doctor_leaves(request: Request, doctor_email: str = Query(..., min_length=5, max_length=254)) -> list[dict[str, Any]]:
    session_user = require_role("doctor")(require_session(request))
    if normalize_email(str(session_user.get("email") or "")) != validate_email(doctor_email):
        raise HTTPException(status_code=403, detail="Doctors can only access their own leave calendar.")
    leaves = dbmod.list_doctor_leaves(get_db(), normalize_email(doctor_email))
    leaves.sort(key=lambda l: l.get("start_at") or "", reverse=True)
    return leaves


@app.post("/api/doctor/leaves")
def upsert_doctor_leave(request: Request, payload: DoctorLeaveRequest) -> dict[str, Any]:
    session_user = require_role("doctor")(require_session(request))
    if normalize_email(str(session_user.get("email") or "")) != validate_email(payload.doctor_email):
        raise HTTPException(status_code=403, detail="Doctors can only update their own leave calendar.")
    doctor_id = normalize_email(payload.doctor_email)
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


@app.post("/appointments/book", response_model=AppointmentResponse)
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
        doctor_id = (payload.doctor_email or "").strip()
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
            "doctor_email": normalize_email(payload.doctor_email) if payload.doctor_email else None,
            "contact_info": contact_phone,
            "priority": None,
            "notes": payload.notes.strip() if payload.notes else None,
            "created_at": created_at,
        },
    )

    return AppointmentResponse(**appointment_to_payload(appointment))


@app.get("/appointments", response_model=list[AppointmentResponse])
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


@app.post("/appointments/cancel", response_model=AppointmentResponse)
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


@app.get("/appointments/dashboard")
def appointments_dashboard(request: Request) -> dict[str, Any]:
    require_role("nurse", "doctor")(require_session(request))
    now = datetime.now(timezone.utc)
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


@app.get("/api/doctor/patients")
def doctor_patients_directory(request: Request) -> list[dict[str, Any]]:
    require_role("doctor")(require_session(request))
    users = load_users()
    patients = [user for user in users if safe_normalize_role(user.get("role") or "patient") == "patient"]
    patients.sort(key=lambda entry: (entry.get("full_name") or "").strip().lower())
    return [user_directory_payload(patient) for patient in patients]


if __name__ == "__main__":
    import uvicorn

    # If you want auto-reload, prefer:
    #   python -m uvicorn backend.api_backend:app --reload --port 8001
    uvicorn.run("backend.api_backend:app", host="127.0.0.1", port=8001, reload=False)
