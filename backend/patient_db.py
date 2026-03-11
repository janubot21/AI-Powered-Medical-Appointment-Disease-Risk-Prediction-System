from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class PatientDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patient_profiles (
                    patient_id TEXT PRIMARY KEY,
                    patient_name TEXT,
                    age REAL,
                    gender TEXT,
                    symptoms TEXT,
                    symptom_count REAL,
                    glucose REAL,
                    height_cm REAL,
                    weight_kg REAL,
                    weight_category TEXT,
                    calculated_bmi REAL,
                    blood_pressure_systolic REAL,
                    blood_pressure_diastolic REAL,
                    smoking_habit TEXT,
                    alcohol_habit TEXT,
                    family_history TEXT,
                    medical_history TEXT,
                    health_data_submitted_at TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            existing_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(patient_profiles)").fetchall()
            }
            if "health_data_submitted_at" not in existing_columns:
                conn.execute("ALTER TABLE patient_profiles ADD COLUMN health_data_submitted_at TEXT")

    def upsert_profile(self, payload: Dict[str, Any]) -> None:
        patient_id = str(payload.get("patient_id", "")).strip()
        if not patient_id:
            return
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO patient_profiles (
                    patient_id, patient_name, age, gender, symptoms, symptom_count, glucose,
                    height_cm, weight_kg, weight_category, calculated_bmi,
                    blood_pressure_systolic, blood_pressure_diastolic, smoking_habit,
                    alcohol_habit, family_history, medical_history, health_data_submitted_at, updated_at
                ) VALUES (
                    :patient_id, :patient_name, :age, :gender, :symptoms, :symptom_count, :glucose,
                    :height_cm, :weight_kg, :weight_category, :calculated_bmi,
                    :blood_pressure_systolic, :blood_pressure_diastolic, :smoking_habit,
                    :alcohol_habit, :family_history, :medical_history, :health_data_submitted_at, :updated_at
                )
                ON CONFLICT(patient_id) DO UPDATE SET
                    patient_name=excluded.patient_name,
                    age=excluded.age,
                    gender=excluded.gender,
                    symptoms=excluded.symptoms,
                    symptom_count=excluded.symptom_count,
                    glucose=excluded.glucose,
                    height_cm=excluded.height_cm,
                    weight_kg=excluded.weight_kg,
                    weight_category=excluded.weight_category,
                    calculated_bmi=excluded.calculated_bmi,
                    blood_pressure_systolic=excluded.blood_pressure_systolic,
                    blood_pressure_diastolic=excluded.blood_pressure_diastolic,
                    smoking_habit=excluded.smoking_habit,
                    alcohol_habit=excluded.alcohol_habit,
                    family_history=excluded.family_history,
                    medical_history=excluded.medical_history,
                    health_data_submitted_at=excluded.health_data_submitted_at,
                    updated_at=excluded.updated_at
                """,
                {
                    "patient_id": patient_id,
                    "patient_name": payload.get("patient_name"),
                    "age": payload.get("age"),
                    "gender": payload.get("gender"),
                    "symptoms": payload.get("symptoms"),
                    "symptom_count": payload.get("symptom_count"),
                    "glucose": payload.get("glucose"),
                    "height_cm": payload.get("height_cm"),
                    "weight_kg": payload.get("weight_kg"),
                    "weight_category": payload.get("weight_category"),
                    "calculated_bmi": payload.get("calculated_bmi"),
                    "blood_pressure_systolic": payload.get("blood_pressure_systolic"),
                    "blood_pressure_diastolic": payload.get("blood_pressure_diastolic"),
                    "smoking_habit": payload.get("smoking_habit"),
                    "alcohol_habit": payload.get("alcohol_habit"),
                    "family_history": payload.get("family_history"),
                    "medical_history": payload.get("medical_history"),
                    "health_data_submitted_at": payload.get("health_data_submitted_at") or now,
                    "updated_at": now,
                },
            )

    def list_profiles(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    patient_id,
                    patient_name,
                    age,
                    gender,
                    symptoms,
                    symptom_count,
                    glucose,
                    height_cm,
                    weight_kg,
                    weight_category,
                    calculated_bmi,
                    blood_pressure_systolic,
                    blood_pressure_diastolic,
                    smoking_habit,
                    alcohol_habit,
                    family_history,
                    medical_history,
                    health_data_submitted_at,
                    updated_at
                FROM patient_profiles
                ORDER BY
                    CASE
                        WHEN patient_id GLOB '[0-9]*' THEN CAST(patient_id AS INTEGER)
                        ELSE NULL
                    END ASC,
                    patient_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def weight_category_from_bmi(bmi: Optional[float]) -> str:
        if bmi is None:
            return ""
        if bmi < 18.5:
            return "Underweight"
        if bmi < 25:
            return "Normal"
        if bmi < 30:
            return "Overweight"
        return "Obese"
