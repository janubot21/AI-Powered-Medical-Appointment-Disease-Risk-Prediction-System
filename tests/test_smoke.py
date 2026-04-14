import importlib
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone


def build_client():
    tmpdir = tempfile.TemporaryDirectory()
    os.environ["APP_ENV"] = "development"
    os.environ["APP_SESSION_SECRET"] = "test-secret-change-me-please-32chars"
    os.environ["APP_DB_PATH"] = os.path.join(tmpdir.name, "test.db")

    import backend.config  # noqa: WPS433
    import backend.services.shared_core  # noqa: WPS433
    import backend.services.shared_domain  # noqa: WPS433
    import backend.services.shared_models  # noqa: WPS433
    import backend.services.shared  # noqa: WPS433
    import backend.api_backend  # noqa: WPS433

    importlib.reload(backend.config)
    importlib.reload(backend.services.shared_core)
    importlib.reload(backend.services.shared_models)
    importlib.reload(backend.services.shared_domain)
    importlib.reload(backend.services.shared)
    importlib.reload(backend.api_backend)

    from fastapi.testclient import TestClient  # noqa: WPS433

    return TestClient(backend.api_backend.app), tmpdir, backend.api_backend


class SmokeTests(unittest.TestCase):
    def test_basic_flows(self):
        client, tmpdir, backend_api = build_client()
        try:
            res = client.get("/health")
            self.assertEqual(res.status_code, 200)

            # Root page should resolve to the role-selection HTML page.
            res = client.get("/")
            self.assertEqual(res.status_code, 200, res.text)

            # Register a fresh patient user.
            token = uuid.uuid4().hex[:8]
            email = f"smoke_{token}@example.com"
            digits = str(uuid.uuid4().int)[-10:]
            phone = digits
            password = "StrongPass1!"

            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Smoke Patient",
                    "email": email,
                    "phone": phone,
                    "password": password,
                    "role": "patient",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            headers = {"X-CSRF-Token": csrf}

            # Session user lookup.
            res = client.get("/auth/user", params={"email": email})
            self.assertEqual(res.status_code, 200, res.text)

            # Submit health details (required before booking UX flows).
            res = client.post(
                "/auth/health-details",
                headers=headers,
                json={
                    "email": email,
                    "age": 30,
                    "gender": "Female",
                    "symptoms": "cough, fatigue",
                    "symptom_count": 2,
                    "glucose": None,
                    "blood_pressure": None,
                    "height_cm": 165,
                    "weight_kg": 60,
                    "bmi": 22.0,
                    "smoking_habit": "No",
                    "alcohol_habit": "No",
                    "average_sleep_hours": 7.5,
                    "medical_history": "",
                    "family_history_major_chronic_disease": "No",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            # Authenticated prediction.
            res = client.post(
                "/predict-risk",
                headers=headers,
                json={
                    "age": 30,
                    "gender": "Female",
                    "symptoms": ["cough", "fatigue"],
                    "glucose": 95.0,
                    "glucose_type": "fasting",
                    "systolic_bp": 120.0,
                    "diastolic_bp": 80.0,
                    "bmi": 22.0,
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            self.assertIn("risk_group", res.json())

            # Booking should work with session + CSRF.
            scheduled_for = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
            res = client.post(
                "/appointments/book",
                headers=headers,
                json={
                    "patient_email": email,
                    "scheduled_for": scheduled_for,
                    "appointment_type": "General checkup",
                    "contact_info": phone,
                    "doctor_email": None,
                    "reason": "Routine",
                    "notes": "",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            appt_id = res.json().get("id")
            self.assertTrue(appt_id)

            # Logout should require CSRF, and succeed.
            res = client.post("/auth/logout", headers=headers)
            self.assertEqual(res.status_code, 200, res.text)
        finally:
            try:
                client.close()
            except Exception:
                pass
            # Ensure SQLite file handles are released on Windows.
            try:
                import backend.services.shared_core as shared_core  # noqa: WPS433

                if getattr(shared_core, "DB_CONN", None) is not None:
                    shared_core.DB_CONN.close()
                    shared_core.DB_CONN = None
            except Exception:
                pass
            tmpdir.cleanup()

    def test_directory_marks_doctor_leave(self):
        client, tmpdir, backend_api = build_client()
        try:
            # Register a doctor and create a leave window.
            token = uuid.uuid4().hex[:8]
            doctor_email = f"doctor_{token}@example.com"
            doctor_phone = str(uuid.uuid4().int)[-10:]
            doctor_password = "StrongPass1!"

            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Smoke Doctor",
                    "email": doctor_email,
                    "phone": doctor_phone,
                    "password": doctor_password,
                    "role": "doctor",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            headers = {"X-CSRF-Token": csrf}

            start_at = (datetime.now(timezone.utc) + timedelta(days=1)).replace(microsecond=0)
            end_at = start_at + timedelta(days=2)
            res = client.post(
                "/api/doctor/leaves",
                headers=headers,
                json={
                    "doctor_email": doctor_email,
                    "start_at": start_at.isoformat(),
                    "end_at": end_at.isoformat(),
                    "reason": "Test leave",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            # Switch session to a patient (directory endpoint requires a session).
            res = client.post("/auth/logout", headers=headers)
            self.assertEqual(res.status_code, 200, res.text)

            patient_email = f"patient_{token}@example.com"
            patient_phone = str(uuid.uuid4().int)[-10:]
            patient_password = "StrongPass1!"
            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Smoke Patient",
                    "email": patient_email,
                    "phone": patient_phone,
                    "password": patient_password,
                    "role": "patient",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            scheduled_for = (start_at + timedelta(hours=1)).isoformat()
            res = client.get("/portal/directory", params={"scheduled_for": scheduled_for})
            self.assertEqual(res.status_code, 200, res.text)
            data = res.json()
            doctors = data.get("doctors") or []
            match = next((d for d in doctors if (d.get("email") or "").lower() == doctor_email.lower()), None)
            self.assertIsNotNone(match)
            self.assertTrue(match.get("on_leave"))
            self.assertFalse(match.get("available"))
            self.assertIsInstance(match.get("leave"), dict)
        finally:
            try:
                client.close()
            except Exception:
                pass
            # Ensure SQLite file handles are released on Windows.
            try:
                import backend.services.shared_core as shared_core  # noqa: WPS433

                if getattr(shared_core, "DB_CONN", None) is not None:
                    shared_core.DB_CONN.close()
                    shared_core.DB_CONN = None
            except Exception:
                pass
            tmpdir.cleanup()

    def test_doctor_dashboard_filters_assigned_appointments(self):
        client, tmpdir, backend_api = build_client()
        try:
            token = uuid.uuid4().hex[:8]
            doctor1_email = f"doctor1_{token}@example.com"
            doctor2_email = f"doctor2_{token}@example.com"
            password = "StrongPass1!"

            # Create two doctors (logout between registrations to switch sessions).
            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Doctor One",
                    "email": doctor1_email,
                    "phone": str(uuid.uuid4().int)[-10:],
                    "password": password,
                    "role": "doctor",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            res = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
            self.assertEqual(res.status_code, 200, res.text)

            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Doctor Two",
                    "email": doctor2_email,
                    "phone": str(uuid.uuid4().int)[-10:],
                    "password": password,
                    "role": "doctor",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            res = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
            self.assertEqual(res.status_code, 200, res.text)

            # Create a patient session and book 3 appointments: assigned to doctor1, assigned to doctor2, unassigned.
            patient_email = f"patient_{token}@example.com"
            patient_phone = str(uuid.uuid4().int)[-10:]
            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Smoke Patient",
                    "email": patient_email,
                    "phone": patient_phone,
                    "password": password,
                    "role": "patient",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            headers = {"X-CSRF-Token": csrf}

            scheduled_for = (datetime.now(timezone.utc) + timedelta(days=3)).isoformat()
            res = client.post(
                "/appointments/book",
                headers=headers,
                json={
                    "patient_email": patient_email,
                    "scheduled_for": scheduled_for,
                    "appointment_type": "General checkup",
                    "contact_info": patient_phone,
                    "doctor_email": doctor1_email,
                    "reason": "Assigned to doctor1",
                    "notes": "",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            res = client.post(
                "/appointments/book",
                headers=headers,
                json={
                    "patient_email": patient_email,
                    "scheduled_for": (datetime.now(timezone.utc) + timedelta(days=4)).isoformat(),
                    "appointment_type": "General checkup",
                    "contact_info": patient_phone,
                    "doctor_email": doctor2_email,
                    "reason": "Assigned to doctor2",
                    "notes": "",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            res = client.post(
                "/appointments/book",
                headers=headers,
                json={
                    "patient_email": patient_email,
                    "scheduled_for": (datetime.now(timezone.utc) + timedelta(days=5)).isoformat(),
                    "appointment_type": "General checkup",
                    "contact_info": patient_phone,
                    "doctor_email": None,
                    "reason": "Unassigned",
                    "notes": "",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            # Switch session to doctor1 and verify dashboard only returns doctor1's assigned appointments.
            res = client.post("/auth/logout", headers=headers)
            self.assertEqual(res.status_code, 200, res.text)

            res = client.post(
                "/auth/login",
                json={"email": doctor1_email, "password": password, "role": "doctor"},
            )
            self.assertEqual(res.status_code, 200, res.text)

            res = client.get("/appointments/dashboard")
            self.assertEqual(res.status_code, 200, res.text)
            data = res.json() or {}
            upcoming = data.get("upcoming") or []
            past = data.get("past") or []
            self.assertEqual(len(past), 0)
            self.assertEqual(len(upcoming), 1)
            self.assertEqual((upcoming[0].get("doctor_email") or "").lower(), doctor1_email.lower())
        finally:
            try:
                client.close()
            except Exception:
                pass
            # Ensure SQLite file handles are released on Windows.
            try:
                import backend.services.shared_core as shared_core  # noqa: WPS433

                if getattr(shared_core, "DB_CONN", None) is not None:
                    shared_core.DB_CONN.close()
                    shared_core.DB_CONN = None
            except Exception:
                pass
            tmpdir.cleanup()

    def test_portal_directory_only_returns_db_doctors(self):
        client, tmpdir, backend_api = build_client()
        try:
            # Create a patient session (directory endpoint requires any authenticated user).
            token = uuid.uuid4().hex[:8]
            password = "StrongPass1!"
            patient_email = f"patient_dir_{token}@example.com"
            patient_phone = str(uuid.uuid4().int)[-10:]
            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Directory Patient",
                    "email": patient_email,
                    "phone": patient_phone,
                    "password": password,
                    "role": "patient",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            # Register a brand-new doctor (should appear in the directory dropdown list).
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            res = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
            self.assertEqual(res.status_code, 200, res.text)

            doctor_email = f"doctor_dir_{token}@example.com"
            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Directory Doctor",
                    "email": doctor_email,
                    "phone": str(uuid.uuid4().int)[-10:],
                    "password": password,
                    "role": "doctor",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            # Switch back to patient session to call /portal/directory.
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            res = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
            self.assertEqual(res.status_code, 200, res.text)

            res = client.post(
                "/auth/login",
                json={
                    "email": patient_email,
                    "password": password,
                    "role": "patient",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            res = client.get("/portal/directory")
            self.assertEqual(res.status_code, 200, res.text)
            data = res.json() or {}
            doctors = data.get("doctors") or []
            self.assertTrue(doctors)
            self.assertIn(doctor_email.lower(), {(d.get("email") or "").lower() for d in doctors})

            # Ensure every returned doctor corresponds to a real SQLite user with role == "doctor".
            import backend.services.shared_core as shared_core  # noqa: WPS433

            db_doctor_emails = {
                (u.get("email") or "").lower()
                for u in shared_core.get_all_users()
                if (u.get("role") or "").lower() == "doctor"
            }
            for d in doctors:
                self.assertEqual((d.get("role") or "").lower(), "doctor")
                self.assertIn((d.get("email") or "").lower(), db_doctor_emails)
        finally:
            try:
                client.close()
            except Exception:
                pass
            # Ensure SQLite file handles are released on Windows.
            try:
                import backend.services.shared_core as shared_core  # noqa: WPS433

                if getattr(shared_core, "DB_CONN", None) is not None:
                    shared_core.DB_CONN.close()
                    shared_core.DB_CONN = None
            except Exception:
                pass
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
