from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR
DATASETS_DIR = PROJECT_ROOT / "datasets"

def _path_from_env(env_key: str, default_path: Path) -> Path:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default_path
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


NEW_PATIENT_CSV = _path_from_env("NEW_PATIENT_CSV_PATH", DATASETS_DIR / "new_patient_data.csv")
PATIENTS_CSV = _path_from_env("PATIENTS_CSV_PATH", DATASETS_DIR / "patients.csv")
DOCTOR_ACCOUNTS_CSV = _path_from_env("DOCTOR_ACCOUNTS_CSV_PATH", DATASETS_DIR / "doctor_accounts.csv")
DOCTOR_LEAVE_CSV = _path_from_env("DOCTOR_LEAVE_CSV_PATH", DATASETS_DIR / "doctor_leave.csv")
DOCTOR_PROFILE_CSV = _path_from_env("DOCTOR_PROFILE_CSV_PATH", DATASETS_DIR / "doctor_profiles.csv")
PATIENT_ACCOUNTS_CSV = _path_from_env("PATIENT_ACCOUNTS_CSV_PATH", DATASETS_DIR / "patient_accounts.csv")
PATIENT_DB_PATH = _path_from_env("PATIENT_DB_PATH", DATASETS_DIR / "patient_records.db")


def ensure_csv_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required CSV not found: {path}. "
            f"Run `python init_datasets.py` once to create required dataset files."
        )
