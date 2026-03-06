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

from paths import DOCTOR_ACCOUNTS_CSV, ensure_csv_exists

DOCTOR_ACCOUNTS_PATH = DOCTOR_ACCOUNTS_CSV
DOCTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9@._-]+$")
PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
AADHAAR_PATTERN = re.compile(r"^[0-9]{12}$")
PASSWORD_POLICY_PATTERN = re.compile(r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")
PASSWORD_POLICY_MESSAGE = (
    "Password must contain minimum 8 characters, including uppercase (A-Z), lowercase (a-z), number (0-9), and special character (@,!,#,$,%,&,*)."
)


class DoctorSignupRequest(BaseModel):
    doctor_id: str = Field(..., min_length=1, description="New doctor identifier")
    id_type: str = Field(..., min_length=1, description="aadhaar or pan")
    id_number: str = Field(..., min_length=1, description="ID number")
    password: str = Field(..., min_length=8, description="New doctor password")

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
        if not PASSWORD_POLICY_PATTERN.fullmatch(cleaned):
            raise ValueError(PASSWORD_POLICY_MESSAGE)
        return cleaned

    @field_validator("id_type")
    @classmethod
    def validate_id_type(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in {"aadhaar", "pan"}:
            raise ValueError("id_type must be either aadhaar or pan")
        return cleaned

    @field_validator("id_number")
    @classmethod
    def validate_id_number(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("id_number cannot be empty")
        return cleaned


class DoctorLoginRequest(BaseModel):
    doctor_id: str = Field(..., min_length=1, description="Existing doctor identifier")
    id_type: str = Field(..., min_length=1, description="aadhaar or pan")
    id_number: str = Field(..., min_length=1, description="ID number")
    password: str = Field(..., min_length=1, description="Doctor password")

    @field_validator("doctor_id", "password", "id_type", "id_number")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned


class DoctorAuthManager:
    def __init__(self, csv_path: Path = DOCTOR_ACCOUNTS_PATH) -> None:
        self.csv_path = csv_path
        ensure_csv_exists(self.csv_path)

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return digest.hex()

    @staticmethod
    def _normalize_id_type(value: str) -> str:
        raw = value.strip().lower()
        if raw not in {"aadhaar", "pan"}:
            raise ValueError("ID Type must be Aadhaar or PAN.")
        return raw

    @staticmethod
    def _normalize_id_number(id_type: str, value: str) -> str:
        cleaned = value.strip().upper()
        if id_type == "aadhaar":
            if not AADHAAR_PATTERN.fullmatch(cleaned):
                raise ValueError("Aadhaar number must be exactly 12 digits.")
            return cleaned
        if not PAN_PATTERN.fullmatch(cleaned):
            raise ValueError("PAN must be in format: AAAAA9999A.")
        return cleaned

    @classmethod
    def _compose_unique_code(cls, id_type: str, id_number: str) -> str:
        norm_type = cls._normalize_id_type(id_type)
        norm_number = cls._normalize_id_number(norm_type, id_number)
        return f"{norm_type.upper()}:{norm_number}"

    def _read_accounts(self) -> Dict[str, Dict[str, str]]:
        accounts: Dict[str, Dict[str, str]] = {}
        ensure_csv_exists(self.csv_path)

        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                did = str(row.get("doctor_id", "")).strip()
                if not did:
                    continue
                accounts[did] = row
        return accounts

    def signup(self, doctor_id: str, id_type: str, id_number: str, password: str) -> None:
        did = doctor_id.strip()
        ucode = self._compose_unique_code(id_type, id_number)
        pwd = password.strip()
        if not PASSWORD_POLICY_PATTERN.fullmatch(pwd):
            raise ValueError(PASSWORD_POLICY_MESSAGE)
        accounts = self._read_accounts()

        if did in accounts:
            raise ValueError("Doctor ID already exists. Please use login.")
        if any(str(row.get("unique_code", "")).strip().upper() == ucode for row in accounts.values()):
            raise ValueError("Unique Code already exists. Use a different Unique Code.")

        salt = os.urandom(16)
        password_hash = self._hash_password(pwd, salt)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["doctor_id", "unique_code", "salt_hex", "password_hash", "created_at"],
            )
            writer.writerow(
                {
                    "doctor_id": did,
                    "unique_code": ucode,
                    "salt_hex": salt.hex(),
                    "password_hash": password_hash,
                    "created_at": created_at,
                }
            )

    def login(self, doctor_id: str, id_type: str, id_number: str, password: str) -> str:
        did = doctor_id.strip()
        has_did = bool(did)
        has_id_num = bool(id_number.strip())
        pwd = password.strip()
        accounts = self._read_accounts()

        if not has_did and not has_id_num:
            raise ValueError("Enter Doctor ID OR ID Number.")

        ucode = ""
        if has_id_num:
            ucode = self._compose_unique_code(id_type, id_number)

        candidate_rows = []
        if has_did and did in accounts:
            candidate_rows.append(accounts[did])

        if has_id_num:
            raw_id_num = id_number.strip().upper()
            for row in accounts.values():
                stored_ucode = str(row.get("unique_code", "")).strip().upper()
                if stored_ucode in {ucode, raw_id_num}:
                    candidate_rows.append(row)

        if not candidate_rows:
            raise ValueError("Invalid Doctor ID, ID Number, or password.")

        matched_doctor_id = ""
        for row in candidate_rows:
            try:
                salt = bytes.fromhex(str(row.get("salt_hex", "")))
            except ValueError as exc:
                raise ValueError("Stored credential format is invalid.") from exc

            expected_hash = str(row.get("password_hash", ""))
            provided_hash = self._hash_password(pwd, salt)
            if hmac.compare_digest(provided_hash, expected_hash):
                matched_doctor_id = str(row.get("doctor_id", "")).strip()
                break

        if not matched_doctor_id:
            raise ValueError("Invalid Doctor ID, ID Number, or password.")
        return matched_doctor_id
