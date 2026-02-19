from __future__ import annotations

import csv
import hashlib
import hmac
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent
PATIENT_ACCOUNTS_PATH = BASE_DIR / "patient_accounts.csv"


class PatientSignupRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, description="New patient identifier")
    password: str = Field(..., min_length=4, description="New patient password")

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("patient_id cannot be empty")
        return cleaned

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 4:
            raise ValueError("password must be at least 4 characters")
        return cleaned


class PatientLoginRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, description="Existing patient identifier")
    password: str = Field(..., min_length=1, description="Patient password")

    @field_validator("patient_id", "password")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned


class PatientAuthManager:
    def __init__(self, csv_path: Path = PATIENT_ACCOUNTS_PATH) -> None:
        self.csv_path = csv_path
        self._ensure_store_exists()

    def _ensure_store_exists(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["patient_id", "salt_hex", "password_hash", "created_at"],
            )
            writer.writeheader()

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return digest.hex()

    def _read_accounts(self) -> Dict[str, Dict[str, str]]:
        accounts: Dict[str, Dict[str, str]] = {}
        if not self.csv_path.exists():
            return accounts

        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                pid = str(row.get("patient_id", "")).strip()
                if not pid:
                    continue
                accounts[pid] = row
        return accounts

    def signup(self, patient_id: str, password: str) -> None:
        pid = patient_id.strip()
        pwd = password.strip()
        accounts = self._read_accounts()

        if pid in accounts:
            raise ValueError("Patient ID already exists. Please use login.")

        salt = os.urandom(16)
        password_hash = self._hash_password(pwd, salt)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["patient_id", "salt_hex", "password_hash", "created_at"],
            )
            writer.writerow(
                {
                    "patient_id": pid,
                    "salt_hex": salt.hex(),
                    "password_hash": password_hash,
                    "created_at": created_at,
                }
            )

    def login(self, patient_id: str, password: str) -> None:
        pid = patient_id.strip()
        pwd = password.strip()
        accounts = self._read_accounts()

        if pid not in accounts:
            raise ValueError("Invalid Patient ID or password.")

        row = accounts[pid]
        try:
            salt = bytes.fromhex(str(row.get("salt_hex", "")))
        except ValueError as exc:
            raise ValueError("Stored credential format is invalid.") from exc

        expected_hash = str(row.get("password_hash", ""))
        provided_hash = self._hash_password(pwd, salt)

        if not hmac.compare_digest(provided_hash, expected_hash):
            raise ValueError("Invalid Patient ID or password.")
