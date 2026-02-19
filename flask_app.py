from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from doctor_auth import DoctorAuthManager
from patient_auth import PatientAuthManager
from predict import LABEL_ENCODER_PATH, MODEL_PATH, RiskEngine, to_jsonable


BASE_DIR = Path(__file__).resolve().parent
NEW_PATIENT_DATA_CSV = BASE_DIR / "new_patient_data.csv"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")

patient_auth_manager = PatientAuthManager()
doctor_auth_manager = DoctorAuthManager()
risk_engine = RiskEngine(MODEL_PATH, LABEL_ENCODER_PATH)
APPOINTMENTS: list[Dict[str, Any]] = []


def _to_int(value: str, field_name: str, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid integer.") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}.")
    return parsed


def _to_float(value: str, field_name: str, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid number.") from exc
    if parsed < min_value or parsed > max_value:
        raise ValueError(f"{field_name} must be between {min_value} and {max_value}.")
    return parsed


def append_to_new_patient_csv(row: Dict[str, Any], csv_path: Path = NEW_PATIENT_DATA_CSV) -> None:
    new_row_df = pd.DataFrame([row])
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        updated_df = new_row_df
    updated_df.to_csv(csv_path, index=False)


def _normalize_patient_id(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _default_feature_value(feature_name: str) -> Any:
    numeric_defaults = {
        "Age": 45,
        "Symptom_Count": 1,
        "Glucose": 95,
        "BloodPressure": 120,
        "BMI": 24.5,
    }
    categorical_defaults = {
        "Gender": 1,
        "Symptoms": "none",
        "Age_Group": "Adult",
        "BMI_Category": "Normal",
        "BP_Category": "Normal",
    }
    if feature_name in numeric_defaults:
        return numeric_defaults[feature_name]
    if feature_name in categorical_defaults:
        return categorical_defaults[feature_name]
    if feature_name.startswith("SYM_"):
        return 0
    return 0


def _age_group(age: float) -> str:
    if age < 13:
        return "Child"
    if age < 20:
        return "Teen"
    if age < 40:
        return "Adult"
    if age < 60:
        return "Middle_Age"
    return "Senior"


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _bp_category(bp: float) -> str:
    if bp < 80:
        return "Low"
    if bp <= 120:
        return "Normal"
    if bp <= 139:
        return "Elevated"
    return "High"


def _build_features_from_new_patient_row(row: Dict[str, Any]) -> Dict[str, Any]:
    expected_cols = list(getattr(risk_engine.model, "feature_names_in_", []))
    if expected_cols:
        features = {col: _default_feature_value(str(col)) for col in expected_cols}
    else:
        features = {}

    age = float(row.get("Age", 45))
    gender_raw = str(row.get("Gender", "male")).strip().lower()
    gender_map = {"male": 1, "female": 0, "other": -1}
    gender = gender_map.get(gender_raw, 1)
    symptoms = str(row.get("Symptoms", "")).strip()
    symptom_count = int(float(row.get("Symptom_Count", 0)))
    glucose = float(row.get("Glucose", 95))
    blood_pressure = float(row.get("BloodPressure", 120))
    bmi = float(row.get("BMI", 24.5))

    features.update(
        {
            "Age": age,
            "Gender": gender,
            "Symptoms": symptoms,
            "Symptom_Count": symptom_count,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "BMI": bmi,
            "Age_Group": _age_group(age),
            "BMI_Category": _bmi_category(bmi),
            "BP_Category": _bp_category(blood_pressure),
        }
    )

    if symptoms:
        symptom_tokens = [s.strip().lower().replace(" ", "_") for s in symptoms.split(",") if s.strip()]
        for token in symptom_tokens:
            sym_col = f"SYM_{token}"
            if sym_col in features:
                features[sym_col] = 1

    return features


def _load_new_patient_features(patient_id: str) -> Optional[Dict[str, Any]]:
    if not NEW_PATIENT_DATA_CSV.exists():
        return None
    try:
        df = pd.read_csv(NEW_PATIENT_DATA_CSV)
    except Exception:
        return None

    if "Patient_ID" not in df.columns:
        return None

    pid = _normalize_patient_id(patient_id)
    matches = df[df["Patient_ID"].astype(str).map(_normalize_patient_id) == pid]
    if matches.empty:
        return None

    latest_row = matches.iloc[-1].to_dict()
    return _build_features_from_new_patient_row(latest_row)


def get_features_for_patient(patient_id: str) -> Dict[str, Any]:
    # New patient records override training dataset rows for the same Patient_ID.
    new_features = _load_new_patient_features(patient_id)
    if new_features is not None:
        return new_features
    return risk_engine.get_patient_features(patient_id)


@app.route("/")
def home() -> Any:
    return redirect(url_for("role_login"))


@app.route("/login")
def role_login() -> Any:
    return render_template("flask_role_login.html")


@app.route("/patient/signup", methods=["GET", "POST"])
def patient_signup() -> Any:
    errors: list[str] = []
    form_data = {"patient_id": ""}

    if request.method == "POST":
        patient_id = request.form.get("patient_id", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        form_data["patient_id"] = patient_id

        if not patient_id:
            errors.append("Patient ID is required.")
        if len(password) < 4:
            errors.append("Password must be at least 4 characters.")
        if password != confirm_password:
            errors.append("Password and Confirm Password must match.")

        if not errors:
            try:
                patient_auth_manager.signup(patient_id, password)
                return redirect(url_for("health_details", patient_id=patient_id))
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Signup failed: {exc}")

    return render_template("flask_patient_signup.html", errors=errors, form_data=form_data)


@app.route("/patient/health-details", methods=["GET", "POST"])
def health_details() -> Any:
    errors: list[str] = []
    patient_id = request.args.get("patient_id", "").strip() or request.form.get("patient_id", "").strip()
    form_data = {
        "patient_id": patient_id,
        "age": "",
        "gender": "",
        "symptoms": "",
        "symptom_count": "",
        "glucose": "",
        "blood_pressure": "",
        "bmi": "",
    }

    if request.method == "POST":
        form_data.update(
            {
                "age": request.form.get("age", "").strip(),
                "gender": request.form.get("gender", "").strip().lower(),
                "symptoms": request.form.get("symptoms", "").strip(),
                "symptom_count": request.form.get("symptom_count", "").strip(),
                "glucose": request.form.get("glucose", "").strip(),
                "blood_pressure": request.form.get("blood_pressure", "").strip(),
                "bmi": request.form.get("bmi", "").strip(),
            }
        )

        if not patient_id:
            errors.append("Patient ID is missing. Please signup again.")
        if form_data["gender"] not in {"male", "female", "other"}:
            errors.append("Gender must be one of: male, female, other.")
        if not form_data["symptoms"]:
            errors.append("Symptoms are required.")

        try:
            age = _to_int(form_data["age"], "Age", 0, 130)
            symptom_count = _to_int(form_data["symptom_count"], "Symptom_Count", 0, 100)
            glucose = _to_float(form_data["glucose"], "Glucose", 0, 1000)
            blood_pressure = _to_float(form_data["blood_pressure"], "BloodPressure", 0, 400)
            bmi = _to_float(form_data["bmi"], "BMI", 0, 120)
        except ValueError as exc:
            errors.append(str(exc))
            age = symptom_count = 0
            glucose = blood_pressure = bmi = 0.0

        if not errors:
            row = {
                "Patient_ID": patient_id,
                "Age": age,
                "Gender": form_data["gender"],
                "Symptoms": form_data["symptoms"],
                "Symptom_Count": symptom_count,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "BMI": bmi,
            }
            try:
                append_to_new_patient_csv(row)
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Could not save health details: {exc}")

            if not errors:
                session["health_confirmation"] = row
                return redirect(url_for("health_confirmation"))

    return render_template("flask_health_details.html", errors=errors, form_data=form_data)


@app.route("/patient/health-confirmation")
def health_confirmation() -> Any:
    row = session.get("health_confirmation")
    if not row:
        return redirect(url_for("patient_signup"))
    return render_template("flask_health_confirmation.html", row=row)


@app.route("/patient/login", methods=["GET", "POST"])
def patient_login() -> Any:
    errors: list[str] = []
    form_data = {"patient_id": ""}

    if request.method == "POST":
        patient_id = request.form.get("patient_id", "").strip()
        password = request.form.get("password", "").strip()
        form_data["patient_id"] = patient_id

        if not patient_id:
            errors.append("Patient ID is required.")
        if not password:
            errors.append("Password is required.")

        if not errors:
            try:
                patient_auth_manager.login(patient_id, password)
                session["patient_id"] = patient_id
                session.pop("health_confirmation", None)
                return redirect(url_for("book_appointment"))
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Login failed: {exc}")

    return render_template("flask_patient_login.html", errors=errors, form_data=form_data)


@app.route("/patient/book-appointment")
def book_appointment() -> Any:
    patient_id = session.get("patient_id")
    if not patient_id:
        return redirect(url_for("patient_login"))
    return render_template("flask_book_appointment.html", patient_id=patient_id)


@app.get("/patient/features/<patient_id>")
def patient_features(patient_id: str) -> Any:
    try:
        features = get_features_for_patient(patient_id)
        return jsonify({"patient_id": patient_id, "patient_features": features})
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 404
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Patient lookup failed: {exc}"}), 500


@app.post("/patient/book-appointment-submit")
def submit_appointment() -> Any:
    payload = request.get_json(silent=True) or {}
    patient_id = str(payload.get("patient_id", "")).strip()
    doctor_id = str(payload.get("doctor_id", "")).strip()
    appointment_time = str(payload.get("appointment_time", "")).strip()
    patient_features = payload.get("patient_features")

    if not patient_id:
        return jsonify({"detail": "patient_id is required"}), 400
    if not doctor_id:
        return jsonify({"detail": "doctor_id is required"}), 400
    if not appointment_time:
        return jsonify({"detail": "appointment_time is required"}), 400

    try:
        parsed_time = datetime.fromisoformat(appointment_time)
    except ValueError:
        return jsonify({"detail": "appointment_time must be ISO format"}), 400

    if patient_features is None:
        try:
            patient_features = get_features_for_patient(patient_id)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400

    try:
        risk_assessment = risk_engine.predict(patient_features)
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Risk assessment failed: {exc}"}), 500

    result = {
        "booking_status": "confirmed",
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "appointment_time": parsed_time.isoformat(),
    }
    APPOINTMENTS.append(
        {
            "booked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "appointment_time": parsed_time.isoformat(),
            "patient_features": to_jsonable(patient_features),
            "risk_assessment": risk_assessment.model_dump(),
        }
    )
    return jsonify(result)


@app.route("/doctor/signup", methods=["GET", "POST"])
def doctor_signup() -> Any:
    errors: list[str] = []
    form_data = {"doctor_id": ""}

    if request.method == "POST":
        doctor_id = request.form.get("doctor_id", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        form_data["doctor_id"] = doctor_id

        if not doctor_id:
            errors.append("Doctor ID is required.")
        if len(password) < 4:
            errors.append("Password must be at least 4 characters.")
        if password != confirm_password:
            errors.append("Password and Confirm Password must match.")

        if not errors:
            try:
                doctor_auth_manager.signup(doctor_id, password)
                return redirect(url_for("doctor_login"))
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Signup failed: {exc}")

    return render_template("flask_doctor_signup.html", errors=errors, form_data=form_data)


@app.route("/doctor/login", methods=["GET", "POST"])
def doctor_login() -> Any:
    errors: list[str] = []
    form_data = {"doctor_id": ""}

    if request.method == "POST":
        doctor_id = request.form.get("doctor_id", "").strip()
        password = request.form.get("password", "").strip()
        form_data["doctor_id"] = doctor_id

        if not doctor_id:
            errors.append("Doctor ID is required.")
        if not password:
            errors.append("Password is required.")

        if not errors:
            try:
                doctor_auth_manager.login(doctor_id, password)
                session["doctor_id"] = doctor_id
                return redirect(url_for("doctor_dashboard"))
            except ValueError as exc:
                errors.append(str(exc))
            except Exception as exc:  # pragma: no cover - guard
                errors.append(f"Login failed: {exc}")

    return render_template("flask_doctor_login.html", errors=errors, form_data=form_data)


@app.route("/doctor/dashboard")
def doctor_dashboard() -> Any:
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        return redirect(url_for("doctor_login"))
    return render_template("flask_doctor_dashboard.html", doctor_id=doctor_id)


@app.get("/doctor/appointments")
def doctor_appointments() -> Any:
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        return jsonify({"detail": "Unauthorized"}), 401
    return jsonify({"appointments": APPOINTMENTS})


@app.post("/doctor/predict-risk")
def doctor_predict_risk() -> Any:
    doctor_id = session.get("doctor_id")
    if not doctor_id:
        return jsonify({"detail": "Unauthorized"}), 401

    payload = request.get_json(silent=True) or {}
    patient_features = payload.get("patient_features")
    if not patient_features:
        return jsonify({"detail": "patient_features is required"}), 400

    try:
        result = risk_engine.predict(patient_features)
        return jsonify(result.model_dump())
    except ValueError as exc:
        return jsonify({"detail": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - guard
        return jsonify({"detail": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
