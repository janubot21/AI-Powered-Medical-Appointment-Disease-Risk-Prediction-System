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

from paths import PATIENT_ACCOUNTS_CSV, ensure_csv_exists

PATIENT_ACCOUNTS_PATH = PATIENT_ACCOUNTS_CSV
PASSWORD_POLICY_PATTERN = re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")
PASSWORD_POLICY_MESSAGE = (
    "Password must contain minimum 8 characters, including uppercase (A-Z), lowercase (a-z), number (0-9), and special character (@,!,#,$,%,&,*)."
)


class PatientSignupRequest(BaseModel):
    patient_id: str = Field(..., min_length=1, description="New patient identifier")
    password: str = Field(..., min_length=8, description="New patient password")

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
        if not PASSWORD_POLICY_PATTERN.fullmatch(cleaned):
            raise ValueError(PASSWORD_POLICY_MESSAGE)
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
        ensure_csv_exists(self.csv_path)

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return digest.hex()

    def _read_accounts(self) -> Dict[str, Dict[str, str]]:
        accounts: Dict[str, Dict[str, str]] = {}
        ensure_csv_exists(self.csv_path)

        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                pid = str(row.get("patient_id", "")).strip()
                if not pid:
                    continue
                accounts[pid] = row
        return accounts

    def account_exists(self, patient_id: str) -> bool:
        pid = patient_id.strip()
        if not pid:
            return False
        return pid in self._read_accounts()

    def import_plaintext_password(self, patient_id: str, password: str) -> None:
        pid = patient_id.strip()
        pwd = password.strip()
        if not pid or not pwd:
            raise ValueError("patient_id and password are required for migration.")
        if self.account_exists(pid):
            return

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

    def signup(self, patient_id: str, password: str) -> None:
        pid = patient_id.strip()
        pwd = password.strip()
        if not PASSWORD_POLICY_PATTERN.fullmatch(pwd):
            raise ValueError(PASSWORD_POLICY_MESSAGE)
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
