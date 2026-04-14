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

from fastapi import Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # Allow `python backend/services/shared.py` to work by ensuring project root is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.risk_engine import engine
from backend.config import get_settings
from backend import db as dbmod


BASE_DIR = Path(__file__).resolve().parents[1]  # backend/
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

NO_STORE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}
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


PHONE_CLEAN_RE = re.compile(r"\D")
PHONE_VALID_RE = re.compile(r"^\d{10}$")
EMAIL_VALID_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
FULL_NAME_VALID_RE = re.compile(r"^[A-Za-z]+(?: [A-Za-z]+)*$")


def normalize_phone(phone: str) -> str:
    raw = (phone or "").strip()
    cleaned = PHONE_CLEAN_RE.sub("", raw)
    if not cleaned or not PHONE_VALID_RE.match(cleaned):
        raise HTTPException(status_code=400, detail="Phone number must be exactly 10 digits.")
    return cleaned


def validate_email(email: str) -> str:
    normalized = normalize_email(email)
    if not normalized or len(normalized) > 254 or not EMAIL_VALID_RE.match(normalized):
        raise HTTPException(status_code=400, detail="Invalid email format.")
    return normalized


def validate_full_name(full_name: str) -> str:
    normalized = " ".join(str(full_name or "").strip().split())
    if len(normalized) < 2 or len(normalized) > 100:
        raise HTTPException(status_code=400, detail="Full name must be 2-100 characters long.")
    if not FULL_NAME_VALID_RE.match(normalized):
        raise HTTPException(status_code=400, detail="Full name must contain only letters and spaces.")
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


def parse_iso_datetime(value: str) -> datetime:
    # Accept "Z" suffix and naive strings; store normalized UTC.
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


