from __future__ import annotations

import csv
import hashlib
import hmac
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent
DOCTOR_ACCOUNTS_PATH = BASE_DIR / "doctor_accounts.csv"
DOCTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9@._-]+$")


class DoctorSignupRequest(BaseModel):
    doctor_id: str = Field(..., min_length=1, description="New doctor identifier")
    password: str = Field(..., min_length=4, description="New doctor password")

    @field_validator("doctor_id")
    @classmethod
    def validate_doctor_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("doctor_id cannot be empty")
        if not DOCTOR_ID_PATTERN.fullmatch(cleaned):
            raise ValueError("doctor_id may contain letters, numbers, @, ., _, -")
        return cleaned

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned) < 4:
            raise ValueError("password must be at least 4 characters")
        return cleaned


class DoctorLoginRequest(BaseModel):
    doctor_id: str = Field(..., min_length=1, description="Existing doctor identifier")
    password: str = Field(..., min_length=1, description="Doctor password")

    @field_validator("doctor_id", "password")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned


class DoctorAuthManager:
    def __init__(self, csv_path: Path = DOCTOR_ACCOUNTS_PATH) -> None:
        self.csv_path = csv_path
        self._ensure_store_exists()

    def _ensure_store_exists(self) -> None:
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["doctor_id", "salt_hex", "password_hash", "created_at"],
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
                did = str(row.get("doctor_id", "")).strip()
                if not did:
                    continue
                accounts[did] = row
        return accounts

    def signup(self, doctor_id: str, password: str) -> None:
        did = doctor_id.strip()
        pwd = password.strip()
        accounts = self._read_accounts()

        if did in accounts:
            raise ValueError("Doctor ID already exists. Please use login.")

        salt = os.urandom(16)
        password_hash = self._hash_password(pwd, salt)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["doctor_id", "salt_hex", "password_hash", "created_at"],
            )
            writer.writerow(
                {
                    "doctor_id": did,
                    "salt_hex": salt.hex(),
                    "password_hash": password_hash,
                    "created_at": created_at,
                }
            )

    def login(self, doctor_id: str, password: str) -> None:
        did = doctor_id.strip()
        pwd = password.strip()
        accounts = self._read_accounts()

        if did not in accounts:
            raise ValueError("Invalid Doctor ID or password.")

        row = accounts[did]
        try:
            salt = bytes.fromhex(str(row.get("salt_hex", "")))
        except ValueError as exc:
            raise ValueError("Stored credential format is invalid.") from exc

        expected_hash = str(row.get("password_hash", ""))
        provided_hash = self._hash_password(pwd, salt)

        if not hmac.compare_digest(provided_hash, expected_hash):
            raise ValueError("Invalid Doctor ID or password.")
