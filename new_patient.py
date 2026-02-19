from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator


OUTPUT_CSV_PATH = Path(__file__).resolve().parent / "new_patient_data.csv"


class NewPatientFeatures(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    age: int = Field(..., ge=0, le=130)
    gender: str = Field(..., description="male/female/other")
    symptoms: str = Field(..., description="Comma-separated symptoms")
    symptom_count: int = Field(..., ge=0)
    disease: str = Field(..., description="Known or suspected disease")
    glucose: float = Field(..., ge=0)
    blood_pressure: float = Field(..., ge=0)
    bmi: float = Field(..., ge=0)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, value: str) -> str:
        allowed = {"male", "female", "other"}
        normalized = value.strip().lower()
        if normalized not in allowed:
            raise ValueError("gender must be one of: male, female, other")
        return normalized

    def to_model_features(self) -> Dict[str, Any]:
        gender_map = {"male": 1, "female": 0, "other": -1}
        return {
            "Age": self.age,
            "Gender": gender_map[self.gender],
            "Symptoms": self.symptoms,
            "Symptom_Count": self.symptom_count,
            "Disease": self.disease,
            "Glucose": self.glucose,
            "BloodPressure": self.blood_pressure,
            "BMI": self.bmi,
        }

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "Patient_ID": self.patient_id,
            "Age": self.age,
            "Gender": self.gender,
            "Symptoms": self.symptoms,
            "Symptom_Count": self.symptom_count,
            "Disease": self.disease,
            "Glucose": self.glucose,
            "BloodPressure": self.blood_pressure,
            "BMI": self.bmi,
        }


def collect_new_patient() -> NewPatientFeatures:
    return NewPatientFeatures(
        patient_id=input("Patient ID: ").strip(),
        age=int(input("Age: ").strip()),
        gender=input("Gender (male/female/other): ").strip(),
        symptoms=input("Symptoms (comma-separated): ").strip(),
        symptom_count=int(input("Symptoms Count: ").strip()),
        disease=input("Disease: ").strip(),
        glucose=float(input("Glucose: ").strip()),
        blood_pressure=float(input("Blood Pressure: ").strip()),
        bmi=float(input("BMI: ").strip()),
    )


def append_patient_to_csv(patient: NewPatientFeatures, csv_path: Path = OUTPUT_CSV_PATH) -> Path:
    row = patient.to_csv_row()
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return csv_path


if __name__ == "__main__":
    patient = collect_new_patient()
    saved_path = append_patient_to_csv(patient)
    features = patient.to_model_features()

    print(f"\nPatient data saved to CSV: {saved_path}")
    print("Prepared patient features for prediction:")
    print(features)
