from __future__ import annotations

import csv
from pathlib import Path

from paths import (
    DATASETS_DIR,
    DOCTOR_ACCOUNTS_CSV,
    NEW_PATIENT_CSV,
    PATIENT_ACCOUNTS_CSV,
    PATIENTS_CSV,
)


def _ensure_csv(path: Path, headers: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_csv(
        NEW_PATIENT_CSV,
        [
            "Patient_ID",
            "Age",
            "Gender",
            "Symptoms",
            "Symptom_Count",
            "Glucose",
            "BloodPressure",
            "BMI",
            "Height_cm",
            "Weight_kg",
            "Smoking_Habit",
            "Alcohol_Habit",
            "Medical_History",
            "Family_History",
            "Health_Data_Submitted_At",
        ],
    )
    _ensure_csv(
        PATIENTS_CSV,
        ["patient_id", "name", "unique_code", "password", "health_details_submitted", "created_at"],
    )
    _ensure_csv(
        DOCTOR_ACCOUNTS_CSV,
        ["doctor_id", "unique_code", "salt_hex", "password_hash", "created_at"],
    )
    _ensure_csv(
        PATIENT_ACCOUNTS_CSV,
        ["patient_id", "salt_hex", "password_hash", "created_at"],
    )

    print(f"Initialized dataset CSVs in: {DATASETS_DIR}")


if __name__ == "__main__":
    main()
