from __future__ import annotations

from contextlib import closing
import json
import sqlite3
from datetime import datetime, timezone
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
        with closing(self._connect()) as conn:
            with conn:
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
                        average_sleep_hours REAL,
                        family_history TEXT,
                        medical_history TEXT,
                        health_data_submitted_at TEXT,
                        nurse_updated_by TEXT,
                        nurse_notes TEXT,
                        doctor_reviewed_by TEXT,
                        reviewed_at TEXT,
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
                if "average_sleep_hours" not in existing_columns:
                    conn.execute("ALTER TABLE patient_profiles ADD COLUMN average_sleep_hours REAL")
                if "nurse_updated_by" not in existing_columns:
                    conn.execute("ALTER TABLE patient_profiles ADD COLUMN nurse_updated_by TEXT")
                if "nurse_notes" not in existing_columns:
                    conn.execute("ALTER TABLE patient_profiles ADD COLUMN nurse_notes TEXT")
                if "doctor_reviewed_by" not in existing_columns:
                    conn.execute("ALTER TABLE patient_profiles ADD COLUMN doctor_reviewed_by TEXT")
                if "reviewed_at" not in existing_columns:
                    conn.execute("ALTER TABLE patient_profiles ADD COLUMN reviewed_at TEXT")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS appointments (
                        appointment_id TEXT PRIMARY KEY,
                        booked_at TEXT NOT NULL,
                        patient_id TEXT NOT NULL,
                        patient_name TEXT,
                        contact_info TEXT,
                        doctor_id TEXT NOT NULL,
                        appointment_type TEXT,
                        appointment_time TEXT NOT NULL,
                        patient_features_json TEXT,
                        risk_assessment_json TEXT,
                        appointment_priority TEXT,
                        recommended_slot TEXT,
                        priority_badge_text TEXT,
                        priority_badge_color TEXT
                    )
                    """
                )

    def upsert_profile(self, payload: Dict[str, Any]) -> None:
        patient_id = str(payload.get("patient_id", "")).strip()
        if not patient_id:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO patient_profiles (
                        patient_id, patient_name, age, gender, symptoms, symptom_count, glucose,
                        height_cm, weight_kg, weight_category, calculated_bmi,
                        blood_pressure_systolic, blood_pressure_diastolic, smoking_habit,
                        alcohol_habit, average_sleep_hours, family_history, medical_history,
                        health_data_submitted_at, nurse_updated_by, nurse_notes,
                        doctor_reviewed_by, reviewed_at, updated_at
                    ) VALUES (
                        :patient_id, :patient_name, :age, :gender, :symptoms, :symptom_count, :glucose,
                        :height_cm, :weight_kg, :weight_category, :calculated_bmi,
                        :blood_pressure_systolic, :blood_pressure_diastolic, :smoking_habit,
                        :alcohol_habit, :average_sleep_hours, :family_history, :medical_history,
                        :health_data_submitted_at, :nurse_updated_by, :nurse_notes,
                        :doctor_reviewed_by, :reviewed_at, :updated_at
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
                        average_sleep_hours=excluded.average_sleep_hours,
                        family_history=excluded.family_history,
                        medical_history=excluded.medical_history,
                        health_data_submitted_at=excluded.health_data_submitted_at,
                        nurse_updated_by=excluded.nurse_updated_by,
                        nurse_notes=excluded.nurse_notes,
                        doctor_reviewed_by=excluded.doctor_reviewed_by,
                        reviewed_at=excluded.reviewed_at,
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
                        "average_sleep_hours": payload.get("average_sleep_hours"),
                        "family_history": payload.get("family_history"),
                        "medical_history": payload.get("medical_history"),
                        "health_data_submitted_at": payload.get("health_data_submitted_at") or now,
                        "nurse_updated_by": payload.get("nurse_updated_by"),
                        "nurse_notes": payload.get("nurse_notes"),
                        "doctor_reviewed_by": payload.get("doctor_reviewed_by"),
                        "reviewed_at": payload.get("reviewed_at"),
                        "updated_at": now,
                    },
                )

    def list_profiles(self) -> List[Dict[str, Any]]:
        with closing(self._connect()) as conn:
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
                    average_sleep_hours,
                    family_history,
                    medical_history,
                    health_data_submitted_at,
                    nurse_updated_by,
                    nurse_notes,
                    doctor_reviewed_by,
                    reviewed_at,
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

    def get_profile(self, patient_id: str) -> Dict[str, Any]:
        patient_id = str(patient_id).strip()
        if not patient_id:
            return {}
        with closing(self._connect()) as conn:
            row = conn.execute(
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
                    average_sleep_hours,
                    family_history,
                    medical_history,
                    health_data_submitted_at,
                    nurse_updated_by,
                    nurse_notes,
                    doctor_reviewed_by,
                    reviewed_at,
                    updated_at
                FROM patient_profiles
                WHERE patient_id = ?
                """,
                (patient_id,),
            ).fetchone()
        return dict(row) if row else {}

    def add_appointment(self, payload: Dict[str, Any]) -> None:
        appointment_id = str(payload.get("appointment_id", "")).strip()
        if not appointment_id:
            return
        with closing(self._connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO appointments (
                        appointment_id,
                        booked_at,
                        patient_id,
                        patient_name,
                        contact_info,
                        doctor_id,
                        appointment_type,
                        appointment_time,
                        patient_features_json,
                        risk_assessment_json,
                        appointment_priority,
                        recommended_slot,
                        priority_badge_text,
                        priority_badge_color
                    ) VALUES (
                        :appointment_id,
                        :booked_at,
                        :patient_id,
                        :patient_name,
                        :contact_info,
                        :doctor_id,
                        :appointment_type,
                        :appointment_time,
                        :patient_features_json,
                        :risk_assessment_json,
                        :appointment_priority,
                        :recommended_slot,
                        :priority_badge_text,
                        :priority_badge_color
                    )
                    """,
                    {
                        "appointment_id": appointment_id,
                        "booked_at": payload.get("booked_at"),
                        "patient_id": payload.get("patient_id"),
                        "patient_name": payload.get("patient_name"),
                        "contact_info": payload.get("contact_info"),
                        "doctor_id": payload.get("doctor_id"),
                        "appointment_type": payload.get("appointment_type"),
                        "appointment_time": payload.get("appointment_time"),
                        "patient_features_json": json.dumps(payload.get("patient_features") or {}),
                        "risk_assessment_json": json.dumps(payload.get("risk_assessment") or {}),
                        "appointment_priority": payload.get("appointment_priority"),
                        "recommended_slot": payload.get("recommended_slot"),
                        "priority_badge_text": payload.get("priority_badge_text"),
                        "priority_badge_color": payload.get("priority_badge_color"),
                    },
                )

    def list_appointments(self, *, patient_id: str | None = None, doctor_id: str | None = None) -> List[Dict[str, Any]]:
        query = """
            SELECT
                appointment_id,
                booked_at,
                patient_id,
                patient_name,
                contact_info,
                doctor_id,
                appointment_type,
                appointment_time,
                patient_features_json,
                risk_assessment_json,
                appointment_priority,
                recommended_slot,
                priority_badge_text,
                priority_badge_color
            FROM appointments
        """
        clauses: list[str] = []
        params: list[Any] = []
        if patient_id:
            clauses.append("patient_id = ?")
            params.append(str(patient_id).strip())
        if doctor_id:
            clauses.append("doctor_id = ?")
            params.append(str(doctor_id).strip())
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY appointment_time DESC, booked_at DESC"

        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()

        items: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["patient_features"] = self._load_json_text(item.pop("patient_features_json", ""))
            item["risk_assessment"] = self._load_json_text(item.pop("risk_assessment_json", ""))
            items.append(item)
        return items

    def delete_appointment(self, appointment_id: str, patient_id: str | None = None) -> bool:
        appointment_id = str(appointment_id).strip()
        if not appointment_id:
            return False
        query = "DELETE FROM appointments WHERE appointment_id = ?"
        params: list[Any] = [appointment_id]
        if patient_id:
            query += " AND patient_id = ?"
            params.append(str(patient_id).strip())
        with closing(self._connect()) as conn:
            with conn:
                cursor = conn.execute(query, params)
                return cursor.rowcount > 0

    @staticmethod
    def _load_json_text(value: Any) -> Dict[str, Any]:
        text = str(value or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

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
