from __future__ import annotations

import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def _write_csv(path: Path, headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()


def _write_csv_rows(path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _drop_modules(*module_names: str) -> None:
    for module_name in module_names:
        sys.modules.pop(module_name, None)


class PatientAuthSmokeTests(unittest.TestCase):
    def test_patient_signup_and_login_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "patient_accounts.csv"
            _write_csv(csv_path, ["patient_id", "salt_hex", "password_hash", "created_at"])

            patient_auth = importlib.import_module("patient_auth")
            manager = patient_auth.PatientAuthManager(csv_path)

            manager.signup("1001", "StrongPass!1")
            manager.login("1001", "StrongPass!1")

            with self.assertRaises(ValueError):
                manager.login("1001", "WrongPass!1")


class PatientDatabaseSmokeTests(unittest.TestCase):
    def test_appointment_roundtrip_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "patient_records.db"
            patient_db_module = importlib.import_module("patient_db")
            database = patient_db_module.PatientDatabase(db_path)

            database.add_appointment(
                {
                    "appointment_id": "appt-1",
                    "booked_at": "2026-01-01T10:00:00Z",
                    "patient_id": "1001",
                    "patient_name": "Test Patient",
                    "contact_info": "9999999999",
                    "doctor_id": "dr-1",
                    "appointment_type": "API Booking",
                    "appointment_time": "2026-01-02T10:00:00",
                    "patient_features": {"Age": 45, "Symptoms": "cough"},
                    "risk_assessment": {"predicted_class": "Low Risk", "risk_level": "low"},
                    "appointment_priority": "Routine",
                    "recommended_slot": "Next available",
                    "priority_badge_text": "Routine",
                    "priority_badge_color": "#008000",
                }
            )

            appointments = database.list_appointments(patient_id="1001")
            self.assertEqual(len(appointments), 1)
            self.assertEqual(appointments[0]["appointment_id"], "appt-1")
            self.assertEqual(appointments[0]["risk_assessment"]["predicted_class"], "Low Risk")

            self.assertTrue(database.delete_appointment("appt-1", patient_id="1001"))
            self.assertEqual(database.list_appointments(patient_id="1001"), [])


class AppConfigSmokeTests(unittest.TestCase):
    def test_flask_secret_key_prefers_env_value(self) -> None:
        with patch.dict(os.environ, {"FLASK_SECRET_KEY": "test-secret-key"}, clear=False):
            app_config = importlib.import_module("app_config")
            app_config = importlib.reload(app_config)
            self.assertEqual(app_config.get_flask_secret_key(), "test-secret-key")

    def test_flask_secret_key_generates_secure_default(self) -> None:
        env = dict(os.environ)
        env.pop("FLASK_SECRET_KEY", None)
        env.pop("SECRET_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            app_config = importlib.import_module("app_config")
            app_config = importlib.reload(app_config)
            generated = app_config.get_flask_secret_key()
            self.assertGreaterEqual(len(generated), 64)
            self.assertNotEqual(generated, "change-this-secret-key")


class FastAPISmokeTests(unittest.TestCase):
    def test_booking_endpoint_persists_to_shared_database(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "datasets"
            new_patient_csv = data_dir / "new_patient_data.csv"
            patients_csv = data_dir / "patients.csv"
            doctor_accounts_csv = data_dir / "doctor_accounts.csv"
            patient_accounts_csv = data_dir / "patient_accounts.csv"
            nurse_accounts_csv = data_dir / "nurse_accounts.csv"
            doctor_leave_csv = data_dir / "doctor_leave.csv"
            doctor_profile_csv = data_dir / "doctor_profiles.csv"
            patient_db_path = data_dir / "patient_records.db"

            _write_csv(
                new_patient_csv,
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
            _write_csv(
                patients_csv,
                ["patient_id", "name", "unique_code", "password", "health_details_submitted", "created_at"],
            )
            _write_csv(doctor_accounts_csv, ["doctor_id", "unique_code", "salt_hex", "password_hash", "created_at"])
            _write_csv(patient_accounts_csv, ["patient_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(nurse_accounts_csv, ["nurse_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(doctor_leave_csv, ["doctor_id", "leave_date", "reason"])
            _write_csv(doctor_profile_csv, ["doctor_id", "doctor_name", "specialization", "is_emergency_available"])

            env_overrides = {
                "NEW_PATIENT_CSV_PATH": str(new_patient_csv),
                "PATIENTS_CSV_PATH": str(patients_csv),
                "DOCTOR_ACCOUNTS_CSV_PATH": str(doctor_accounts_csv),
                "PATIENT_ACCOUNTS_CSV_PATH": str(patient_accounts_csv),
                "NURSE_ACCOUNTS_CSV_PATH": str(nurse_accounts_csv),
                "DOCTOR_LEAVE_CSV_PATH": str(doctor_leave_csv),
                "DOCTOR_PROFILE_CSV_PATH": str(doctor_profile_csv),
                "PATIENT_DB_PATH": str(patient_db_path),
                "FLASK_SECRET_KEY": "test-secret-key",
                "COOKIE_SECURE": "0",
            }

            with patch.dict(os.environ, env_overrides, clear=False):
                _drop_modules(
                    "app_config",
                    "doctor_auth",
                    "env_loader",
                    "flask_app",
                    "main",
                    "nurse_auth",
                    "paths",
                    "patient_auth",
                    "patient_db",
                    "predict",
                )
                importlib.invalidate_caches()

                main = importlib.import_module("main")
                from fastapi.testclient import TestClient

                client = TestClient(main.app)
                signup_response = client.post(
                    "/patient-signup",
                    json={"patient_id": "1001", "password": "StrongPass!1"},
                )
                self.assertEqual(signup_response.status_code, 200, signup_response.text)

                login_response = client.post(
                    "/patient-login",
                    json={"patient_id": "1001", "password": "StrongPass!1"},
                )
                self.assertEqual(login_response.status_code, 200, login_response.text)
                patient_token = login_response.json()["api_token"]

                features = {
                    str(feature_name): main.risk_engine._default_value_for_feature(str(feature_name))
                    for feature_name in getattr(main.risk_engine.model, "feature_names_in_", [])
                }
                features.update(
                    {
                        "Age": 45,
                        "Gender": 1,
                        "Symptoms": "cough",
                        "Symptom_Count": 1,
                        "Glucose": 95,
                        "BloodPressure": 120,
                        "BMI": 24.0,
                    }
                )

                response = client.post(
                    "/book-appointment",
                    json={
                        "patient_id": "1001",
                        "doctor_id": "dr-1",
                        "appointment_time": "2030-01-01T10:00:00",
                        "patient_features": features,
                    },
                    headers={"Authorization": f"Bearer {patient_token}"},
                )
                self.assertEqual(response.status_code, 200, response.text)

                appointments_response = client.get(
                    "/appointments",
                    headers={"Authorization": f"Bearer {patient_token}"},
                )
                self.assertEqual(appointments_response.status_code, 200)
                appointments = appointments_response.json()["appointments"]
                self.assertEqual(len(appointments), 1)
                self.assertEqual(appointments[0]["patient_id"], "1001")
                self.assertEqual(appointments[0]["doctor_id"], "dr-1")
                self.assertIn("appointment_id", appointments[0])

    def test_sensitive_fastapi_endpoints_require_auth_and_enforce_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "datasets"
            new_patient_csv = data_dir / "new_patient_data.csv"
            patients_csv = data_dir / "patients.csv"
            doctor_accounts_csv = data_dir / "doctor_accounts.csv"
            patient_accounts_csv = data_dir / "patient_accounts.csv"
            nurse_accounts_csv = data_dir / "nurse_accounts.csv"
            doctor_leave_csv = data_dir / "doctor_leave.csv"
            doctor_profile_csv = data_dir / "doctor_profiles.csv"
            patient_db_path = data_dir / "patient_records.db"

            _write_csv_rows(
                new_patient_csv,
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
                [
                    {
                        "Patient_ID": "1001",
                        "Age": 45,
                        "Gender": "male",
                        "Symptoms": "cough",
                        "Symptom_Count": 1,
                        "Glucose": 95,
                        "BloodPressure": 120,
                        "BMI": 24.0,
                        "Height_cm": 170,
                        "Weight_kg": 70,
                        "Smoking_Habit": "no",
                        "Alcohol_Habit": "no",
                        "Medical_History": "",
                        "Family_History": "no",
                        "Health_Data_Submitted_At": "2026-01-01T10:00:00Z",
                    }
                ],
            )
            _write_csv(patients_csv, ["patient_id", "name", "unique_code", "health_details_submitted", "created_at"])
            _write_csv(doctor_accounts_csv, ["doctor_id", "unique_code", "salt_hex", "password_hash", "created_at"])
            _write_csv(patient_accounts_csv, ["patient_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(nurse_accounts_csv, ["nurse_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(doctor_leave_csv, ["doctor_id", "leave_date", "reason"])
            _write_csv(doctor_profile_csv, ["doctor_id", "doctor_name", "specialization", "is_emergency_available"])

            env_overrides = {
                "NEW_PATIENT_CSV_PATH": str(new_patient_csv),
                "PATIENTS_CSV_PATH": str(patients_csv),
                "DOCTOR_ACCOUNTS_CSV_PATH": str(doctor_accounts_csv),
                "PATIENT_ACCOUNTS_CSV_PATH": str(patient_accounts_csv),
                "NURSE_ACCOUNTS_CSV_PATH": str(nurse_accounts_csv),
                "DOCTOR_LEAVE_CSV_PATH": str(doctor_leave_csv),
                "DOCTOR_PROFILE_CSV_PATH": str(doctor_profile_csv),
                "PATIENT_DB_PATH": str(patient_db_path),
                "FLASK_SECRET_KEY": "test-secret-key",
                "COOKIE_SECURE": "0",
            }

            with patch.dict(os.environ, env_overrides, clear=False):
                _drop_modules(
                    "api_auth",
                    "app_config",
                    "doctor_auth",
                    "env_loader",
                    "flask_app",
                    "main",
                    "nurse_auth",
                    "paths",
                    "patient_auth",
                    "patient_db",
                    "predict",
                )
                importlib.invalidate_caches()

                main = importlib.import_module("main")
                from fastapi.testclient import TestClient

                client = TestClient(main.app)
                unauthorized = client.get("/appointments")
                self.assertEqual(unauthorized.status_code, 401)

                signup_response = client.post(
                    "/patient-signup",
                    json={"patient_id": "1001", "password": "StrongPass!1"},
                )
                self.assertEqual(signup_response.status_code, 200, signup_response.text)

                login_response = client.post(
                    "/patient-login",
                    json={"patient_id": "1001", "password": "StrongPass!1"},
                )
                self.assertEqual(login_response.status_code, 200, login_response.text)
                patient_token = login_response.json()["api_token"]

                scoped_response = client.get(
                    "/patient-features/1001",
                    headers={"Authorization": f"Bearer {patient_token}"},
                )
                self.assertEqual(scoped_response.status_code, 200, scoped_response.text)

                forbidden_response = client.get(
                    "/patient-features/9999",
                    headers={"Authorization": f"Bearer {patient_token}"},
                )
                self.assertEqual(forbidden_response.status_code, 403, forbidden_response.text)


class AuthContractTests(unittest.TestCase):
    def test_doctor_api_login_accepts_doctor_id_without_optional_id_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "datasets"
            new_patient_csv = data_dir / "new_patient_data.csv"
            patients_csv = data_dir / "patients.csv"
            doctor_accounts_csv = data_dir / "doctor_accounts.csv"
            patient_accounts_csv = data_dir / "patient_accounts.csv"
            nurse_accounts_csv = data_dir / "nurse_accounts.csv"
            doctor_leave_csv = data_dir / "doctor_leave.csv"
            doctor_profile_csv = data_dir / "doctor_profiles.csv"
            patient_db_path = data_dir / "patient_records.db"

            _write_csv(
                new_patient_csv,
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
            _write_csv(patients_csv, ["patient_id", "name", "unique_code", "health_details_submitted", "created_at"])
            _write_csv(doctor_accounts_csv, ["doctor_id", "unique_code", "salt_hex", "password_hash", "created_at"])
            _write_csv(patient_accounts_csv, ["patient_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(nurse_accounts_csv, ["nurse_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(doctor_leave_csv, ["doctor_id", "leave_date", "reason"])
            _write_csv(doctor_profile_csv, ["doctor_id", "doctor_name", "specialization", "is_emergency_available"])

            env_overrides = {
                "NEW_PATIENT_CSV_PATH": str(new_patient_csv),
                "PATIENTS_CSV_PATH": str(patients_csv),
                "DOCTOR_ACCOUNTS_CSV_PATH": str(doctor_accounts_csv),
                "PATIENT_ACCOUNTS_CSV_PATH": str(patient_accounts_csv),
                "NURSE_ACCOUNTS_CSV_PATH": str(nurse_accounts_csv),
                "DOCTOR_LEAVE_CSV_PATH": str(doctor_leave_csv),
                "DOCTOR_PROFILE_CSV_PATH": str(doctor_profile_csv),
                "PATIENT_DB_PATH": str(patient_db_path),
                "FLASK_SECRET_KEY": "test-secret-key",
                "COOKIE_SECURE": "0",
            }

            with patch.dict(os.environ, env_overrides, clear=False):
                _drop_modules(
                    "api_auth",
                    "app_config",
                    "doctor_auth",
                    "env_loader",
                    "flask_app",
                    "main",
                    "nurse_auth",
                    "paths",
                    "patient_auth",
                    "patient_db",
                    "predict",
                )
                importlib.invalidate_caches()

                main = importlib.import_module("main")
                from fastapi.testclient import TestClient

                client = TestClient(main.app)
                signup_response = client.post(
                    "/doctor-signup",
                    json={"doctor_id": "dr-demo", "password": "StrongPass!1"},
                )
                self.assertEqual(signup_response.status_code, 200, signup_response.text)

                login_response = client.post(
                    "/doctor-login",
                    json={"doctor_id": "dr-demo", "password": "StrongPass!1"},
                )
                self.assertEqual(login_response.status_code, 200, login_response.text)
                self.assertEqual(login_response.json()["doctor_id"], "dr-demo")
                self.assertIn("api_token", login_response.json())


class FlaskPortalSecurityTests(unittest.TestCase):
    def test_patient_portal_signup_and_login_use_hashed_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "datasets"
            new_patient_csv = data_dir / "new_patient_data.csv"
            patients_csv = data_dir / "patients.csv"
            doctor_accounts_csv = data_dir / "doctor_accounts.csv"
            patient_accounts_csv = data_dir / "patient_accounts.csv"
            nurse_accounts_csv = data_dir / "nurse_accounts.csv"
            doctor_leave_csv = data_dir / "doctor_leave.csv"
            doctor_profile_csv = data_dir / "doctor_profiles.csv"
            patient_db_path = data_dir / "patient_records.db"

            _write_csv(
                new_patient_csv,
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
            _write_csv(patients_csv, ["patient_id", "name", "unique_code", "health_details_submitted", "created_at"])
            _write_csv(doctor_accounts_csv, ["doctor_id", "unique_code", "salt_hex", "password_hash", "created_at"])
            _write_csv(patient_accounts_csv, ["patient_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(nurse_accounts_csv, ["nurse_id", "salt_hex", "password_hash", "created_at"])
            _write_csv(doctor_leave_csv, ["doctor_id", "leave_date", "reason"])
            _write_csv(doctor_profile_csv, ["doctor_id", "doctor_name", "specialization", "is_emergency_available"])

            env_overrides = {
                "NEW_PATIENT_CSV_PATH": str(new_patient_csv),
                "PATIENTS_CSV_PATH": str(patients_csv),
                "DOCTOR_ACCOUNTS_CSV_PATH": str(doctor_accounts_csv),
                "PATIENT_ACCOUNTS_CSV_PATH": str(patient_accounts_csv),
                "NURSE_ACCOUNTS_CSV_PATH": str(nurse_accounts_csv),
                "DOCTOR_LEAVE_CSV_PATH": str(doctor_leave_csv),
                "DOCTOR_PROFILE_CSV_PATH": str(doctor_profile_csv),
                "PATIENT_DB_PATH": str(patient_db_path),
                "FLASK_SECRET_KEY": "test-secret-key",
                "COOKIE_SECURE": "0",
            }

            with patch.dict(os.environ, env_overrides, clear=False):
                _drop_modules(
                    "api_auth",
                    "app_config",
                    "doctor_auth",
                    "env_loader",
                    "flask_app",
                    "main",
                    "nurse_auth",
                    "paths",
                    "patient_auth",
                    "patient_db",
                    "predict",
                    "routes.common",
                    "routes.patient",
                    "routes.staff",
                )
                importlib.invalidate_caches()

                flask_app = importlib.import_module("flask_app")
                client = flask_app.app.test_client()

                signup_response = client.post(
                    "/patient/signup",
                    data={
                        "name": "Portal User",
                        "password": "StrongPass!1",
                        "confirm_password": "StrongPass!1",
                    },
                    follow_redirects=False,
                )
                self.assertEqual(signup_response.status_code, 302)

                with patients_csv.open("r", newline="", encoding="utf-8") as csv_file:
                    reader = csv.DictReader(csv_file)
                    self.assertNotIn("password", reader.fieldnames or [])
                    patient_rows = list(reader)
                self.assertEqual(len(patient_rows), 1)
                self.assertEqual(patient_rows[0]["name"], "Portal User")

                with patient_accounts_csv.open("r", newline="", encoding="utf-8") as csv_file:
                    account_rows = list(csv.DictReader(csv_file))
                self.assertEqual(len(account_rows), 1)
                self.assertTrue(account_rows[0]["password_hash"])
                self.assertNotEqual(account_rows[0]["password_hash"], "StrongPass!1")

                login_response = client.post(
                    "/patient/login",
                    data={"patient_name": "Portal User", "password": "StrongPass!1"},
                    follow_redirects=True,
                )
                self.assertEqual(login_response.status_code, 200)
                self.assertIn(b"redirect", login_response.data.lower())


if __name__ == "__main__":
    unittest.main()
