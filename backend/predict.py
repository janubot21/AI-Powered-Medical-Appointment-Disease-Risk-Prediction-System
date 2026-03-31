from __future__ import annotations

import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from paths import NEW_PATIENT_CSV

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "decision_tree_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
FEATURE_DATA_PATH = NEW_PATIENT_CSV


class RiskPredictionRequest(BaseModel):
    patient_features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary expected by the trained model.",
    )

    @field_validator("patient_features")
    @classmethod
    def validate_patient_features(cls, value: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("patient_features cannot be empty")
        return value


class AppointmentBookingRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    doctor_id: Optional[str] = Field(None, description="Assigned doctor identifier")
    appointment_time: datetime = Field(..., description="Requested appointment date-time")
    patient_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Feature dictionary expected by the trained model.",
    )

    @field_validator("patient_id")
    @classmethod
    def validate_patient_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("patient_id cannot be empty")
        return cleaned

    @field_validator("doctor_id")
    @classmethod
    def validate_doctor_id(cls, value: Optional[str]) -> Optional[str]:
        cleaned = str(value or "").strip()
        return cleaned or None

    @field_validator("appointment_time")
    @classmethod
    def validate_appointment_time(cls, value: datetime) -> datetime:
        if value <= datetime.now(value.tzinfo):
            raise ValueError("appointment_time must be in the future")
        return value

    @field_validator("patient_features")
    @classmethod
    def validate_optional_patient_features(cls, value: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if value is not None and (not isinstance(value, dict) or not value):
            raise ValueError("patient_features must be a non-empty object when provided")
        return value


class RiskPredictionResponse(BaseModel):
    predicted_class: str
    risk_probability: float
    confidence_breakdown: Dict[str, float]
    risk_level: str
    contributing_factors: List[str] = Field(default_factory=list)
    doctor_note: str = ""
    patient_guidance: str = ""


class AppointmentBookingResponse(BaseModel):
    booking_status: str
    patient_id: str
    doctor_id: Optional[str] = None
    appointment_time: datetime
    risk_assessment: RiskPredictionResponse
    appointment_priority: str
    recommended_slot: str


class RiskEngine:
    LOW_RISK_SYMPTOMS = {
        "cold",
        "cough",
        "mild_constipation",
        "intermittent_abdominal_discomfort",
        "fever"
    }
    # High-priority symptoms that should force a High Risk rule outcome.
    # Include common text variants after normalization (e.g. "chest pain" -> "chest_pain").
    SYMPTOMS = {
        "chestpain",
        "chest_pain",
        "diarrhea",
        "diarrhoea",
        "insomnia",
        "dizziness",
        "low_bp",
    }
    MEDIUM_RISK_SYMPTOMS = {
        "blurred_vision",
        "swelling",
        "depression",
        "sore_throat",
        "joint_pain",
        "anxiety",
        "muscle_pain",
        "appetite_loss",
        "runny_nose",
    }
    HIGH_RISK_KEYWORDS = {
        "chest_pain",
        "chestpain",
        "breathlessness",
        "shortness_of_breath",
        "dizziness",
        "blurred_vision",
        "low_bp",
    }

    def __init__(self, model_path: Path, label_encoder_path: Optional[Path] = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.label_encoder = None

        if label_encoder_path and label_encoder_path.exists():
            self.label_encoder = joblib.load(label_encoder_path)
        self._feature_data_mtime: Optional[float] = None
        self.patient_feature_map = self._load_patient_feature_map(FEATURE_DATA_PATH)
        self._feature_data_mtime = self._get_mtime(FEATURE_DATA_PATH)

    @staticmethod
    def _get_mtime(path: Path) -> Optional[float]:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _refresh_feature_map_if_needed(self) -> None:
        current_mtime = self._get_mtime(FEATURE_DATA_PATH)
        if current_mtime != self._feature_data_mtime:
            self.patient_feature_map = self._load_patient_feature_map(FEATURE_DATA_PATH)
            self._feature_data_mtime = current_mtime

    @staticmethod
    def _normalize_patient_id(value: Any) -> str:
        text = str(value).strip()
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    def _load_patient_feature_map(self, csv_path: Path) -> Dict[str, Dict[str, Any]]:
        if not csv_path.exists():
            return {}
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return {}
        if "Patient_ID" not in df.columns:
            return {}

        expected_cols = list(getattr(self.model, "feature_names_in_", []))
        if not expected_cols:
            return {}

        feature_map: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            pid = self._normalize_patient_id(row["Patient_ID"])
            row_data = row.to_dict()
            features = {col: self._default_value_for_feature(str(col)) for col in expected_cols}

            def as_float(value: Any, default: float) -> float:
                if value is None or pd.isna(value):
                    return default
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return default

            age = as_float(row_data.get("Age"), 45.0)
            gender_raw = str(row_data.get("Gender", "male")).strip().lower()
            gender_map = {"male": 1, "m": 1, "female": 0, "f": 0, "other": -1}
            gender = gender_map.get(gender_raw, 1)
            symptoms = str(row_data.get("Symptoms", "")).strip()
            symptom_count = int(as_float(row_data.get("Symptom_Count", row_data.get("Sympton_Count", 0)), 0.0))
            glucose = as_float(row_data.get("Glucose"), 95.0)
            blood_pressure = as_float(row_data.get("BloodPressure"), 120.0)
            bmi = as_float(row_data.get("BMI"), 24.5)

            derived = {
                "Age": age,
                "Gender": gender,
                "Symptoms": symptoms,
                "Symptom_Count": symptom_count,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "BMI": bmi,
                "Age_Group": "Child" if age < 13 else "Teen" if age < 20 else "Adult" if age < 40 else "Middle_Age" if age < 60 else "Senior",
                "BMI_Category": "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese",
                "BP_Category": "Low" if blood_pressure < 80 else "Normal" if blood_pressure <= 121 else "Elevated" if blood_pressure <= 139 else "High",
            }
            for key, value in derived.items():
                if key in features:
                    features[key] = value

            if symptoms:
                symptom_tokens = [self._normalize_symptom_name(s) for s in re.split(r"[,;|]+", symptoms) if s.strip()]
                for token in symptom_tokens:
                    sym_col = f"SYM_{token}"
                    if sym_col in features:
                        features[sym_col] = 1
            feature_map[pid] = features
        return feature_map

    def get_patient_features(self, patient_id: str) -> Dict[str, Any]:
        self._refresh_feature_map_if_needed()
        pid = self._normalize_patient_id(patient_id)
        if pid not in self.patient_feature_map:
            raise ValueError("Patient_ID not found")
        return self.patient_feature_map[pid]

    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(features)

        if "Sympton_Count" in normalized and "Symptom_Count" not in normalized:
            normalized["Symptom_Count"] = normalized.pop("Sympton_Count")

        if "Gender" in normalized:
            g = normalized["Gender"]
            if isinstance(g, str):
                g_clean = g.strip().lower()
                gender_map = {
                    "male": 1,
                    "m": 1,
                    "female": 0,
                    "f": 0,
                    "other": -1,
                    "unknown": -1,
                }
                if g_clean in gender_map:
                    normalized["Gender"] = gender_map[g_clean]

        # If Symptoms text is present, treat it as source of truth and rebuild SYM_* flags.
        # This prevents stale symptom indicator columns from forcing incorrect high-risk outputs.
        if "Symptoms" in normalized:
            raw_symptoms = normalized.get("Symptoms")
            symptom_tokens: list[str] = []
            if isinstance(raw_symptoms, str):
                symptom_tokens = [
                    self._normalize_symptom_name(s)
                    for s in re.split(r"[,;|]+", raw_symptoms)
                    if s.strip()
                ]

            for key in list(normalized.keys()):
                if str(key).startswith("SYM_"):
                    normalized[key] = 0
            for token in symptom_tokens:
                normalized[f"SYM_{token}"] = 1

        preprocessor = getattr(self.model, "named_steps", {}).get("preprocessor")
        if preprocessor is not None:
            for name, _, cols in getattr(preprocessor, "transformers", []):
                if name != "num":
                    continue
                for col in cols:
                    if col not in normalized:
                        continue
                    value = normalized[col]
                    if isinstance(value, str):
                        v = value.strip()
                        if v == "":
                            continue
                        try:
                            normalized[col] = float(v)
                        except ValueError:
                            continue

        return normalized

    @staticmethod
    def _normalize_symptom_name(value: str) -> str:
        token = value.strip().lower()
        token = token.replace(" ", "_").replace("-", "_")
        token = re.sub(r"[^a-z0-9_]+", "", token)
        return token

    def _extract_symptoms_from_text(self, symptoms_text: str) -> set[str]:
        found: set[str] = set()
        if not symptoms_text.strip():
            return found

        # Keep both delimiter-based tokens and phrase detection for free text inputs.
        raw_parts = [p.strip() for p in re.split(r"[,;|\n]+", symptoms_text) if p.strip()]
        found.update(self._normalize_symptom_name(part) for part in raw_parts)

        cleaned_text = symptoms_text.strip().lower()
        cleaned_text = cleaned_text.replace("-", " ").replace("_", " ")
        cleaned_text = re.sub(r"[^a-z0-9,\s;|]+", " ", cleaned_text)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        known_symptoms = (
            self.SYMPTOMS
            | self.HIGH_RISK_KEYWORDS
            | self.MEDIUM_RISK_SYMPTOMS
            | self.LOW_RISK_SYMPTOMS
        )
        for symptom in known_symptoms:
            normalized = self._normalize_symptom_name(symptom)
            phrase = normalized.replace("_", " ")
            if not phrase:
                continue
            if re.search(rf"\b{re.escape(phrase)}\b", cleaned_text):
                found.add(normalized)

        return found

    def _extract_reported_symptoms(self, features: Dict[str, Any]) -> set[str]:
        found: set[str] = set()

        raw_symptoms = features.get("Symptoms")
        if isinstance(raw_symptoms, str):
            found.update(self._extract_symptoms_from_text(raw_symptoms))

        for key, value in features.items():
            if not str(key).startswith("SYM_"):
                continue
            symptom_name = self._normalize_symptom_name(str(key)[4:])
            try:
                is_present = float(value) == 1.0
            except (TypeError, ValueError):
                is_present = str(value).strip().lower() in {"true", "yes", "y"}
            if is_present:
                found.add(symptom_name)

        return found

    def _rule_based_risk_class(self, features: Dict[str, Any]) -> str:
        def to_number(value: Any, field_name: str) -> float:
            if value is None:
                return 0.0
            if isinstance(value, str):
                value = value.strip()
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field_name} must be a valid number") from exc

        age = to_number(features.get("Age"), "Age")
        symptom_count = to_number(
            features.get("Symptom_Count", features.get("Sympton_Count")),
            "Symptom_Count",
        )
        glucose = to_number(features.get("Glucose"), "Glucose")
        blood_pressure = to_number(features.get("BloodPressure"), "BloodPressure")
        reported_symptoms = self._extract_reported_symptoms(features)
        has_high_risk_symptom = any(sym in reported_symptoms for sym in self.SYMPTOMS | self.HIGH_RISK_KEYWORDS)
        has_medium_symptom = any(sym in reported_symptoms for sym in self.MEDIUM_RISK_SYMPTOMS)

        if has_high_risk_symptom or glucose >= 180 or blood_pressure >= 140 or (age >= 60 and symptom_count >= 3):
            return "High Risk"
        if has_medium_symptom or glucose >= 126 or blood_pressure >= 130 or symptom_count >= 3 or age >= 50:
            return "Medium Risk"
        return "Low Risk"

    @staticmethod
    def _default_value_for_feature(feature_name: str) -> Any:
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

    def _model_based_risk(self, row: pd.DataFrame) -> tuple[str, float]:
        # Model-first scoring; maps model output into a low/medium/high view.
        prediction = self.model.predict(row)[0]

        decoded_label = str(prediction)
        if self.label_encoder is not None:
            try:
                decoded_label = str(self.label_encoder.inverse_transform([int(prediction)])[0])
            except Exception:
                decoded_label = str(prediction)

        decoded_lower = decoded_label.strip().lower()
        if "high" in decoded_lower:
            model_class = "High Risk"
        elif "medium" in decoded_lower:
            model_class = "Medium Risk"
        else:
            model_class = "Low Risk"

        high_prob = 1.0 if model_class == "High Risk" else 0.0
        if hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(row)[0]
                class_names: List[str]
                if self.label_encoder is not None:
                    class_names = [str(c).lower() for c in self.label_encoder.classes_]
                else:
                    model_classes = getattr(self.model, "classes_", [])
                    class_names = [str(c).lower() for c in model_classes]

                high_idx = next((i for i, c in enumerate(class_names) if "high" in c), None)
                medium_idx = next((i for i, c in enumerate(class_names) if "medium" in c), None)
                if high_idx is not None and 0 <= high_idx < len(probs):
                    high_prob = float(probs[high_idx])
                    if model_class == "Medium Risk" and medium_idx is not None and 0 <= medium_idx < len(probs):
                        high_prob = max(high_prob, float(probs[medium_idx]) * 0.65)
                else:
                    if model_class == "High Risk":
                        high_prob = 0.92
                    elif model_class == "Medium Risk":
                        high_prob = 0.62
                    else:
                        high_prob = 0.2
            except Exception:
                if model_class == "High Risk":
                    high_prob = 0.92
                elif model_class == "Medium Risk":
                    high_prob = 0.62
                else:
                    high_prob = 0.2

        return model_class, float(np.clip(high_prob, 0.0, 1.0))

    def _explain_contributing_factors(self, features: Dict[str, Any]) -> List[str]:
        factors: List[str] = []

        def as_float(value: Any) -> Optional[float]:
            try:
                if value is None:
                    return None
                if isinstance(value, str) and value.strip() == "":
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        age = as_float(features.get("Age"))
        glucose = as_float(features.get("Glucose"))
        bp = as_float(features.get("BloodPressure"))
        bmi = as_float(features.get("BMI"))
        symptom_count = as_float(features.get("Symptom_Count", features.get("Sympton_Count")))

        if age is not None and age >= 60:
            factors.append("Age above 60 increased risk priority.")
        elif age is not None and age >= 50:
            factors.append("Age above 50 contributed to moderate risk.")

        if glucose is not None and glucose >= 180:
            factors.append("Very high glucose level was detected.")
        elif glucose is not None and glucose >= 126:
            factors.append("Elevated glucose level contributed to risk.")

        if bp is not None and bp >= 140:
            factors.append("High systolic blood pressure was observed.")
        elif bp is not None and bp >= 130:
            factors.append("Borderline high blood pressure was observed.")

        if bmi is not None and bmi >= 30:
            factors.append("BMI in obese range increased risk.")
        elif bmi is not None and bmi >= 25:
            factors.append("BMI in overweight range contributed to risk.")

        if symptom_count is not None and symptom_count >= 3:
            factors.append("Multiple symptoms were reported.")

        symptoms = self._extract_reported_symptoms(features)
        if symptoms:
            sorted_symptoms = ", ".join(sorted(symptoms)[:4])
            factors.append(f"Reported symptoms considered: {sorted_symptoms}.")

        smoking = str(features.get("Smoking_Habit", "")).strip().lower()
        if smoking in {"yes", "true", "1"}:
            factors.append("Smoking habit increased predicted risk.")
        alcohol = str(features.get("Alcohol_Habit", "")).strip().lower()
        if alcohol in {"yes", "true", "1"}:
            factors.append("Alcohol habit contributed to risk profile.")

        history = str(features.get("Medical_History", "")).strip()
        if history:
            factors.append("Medical history was included in risk estimation.")
        family = str(features.get("Family_History", "")).strip().lower()
        if family in {"yes", "true", "1"}:
            factors.append("Positive family history increased baseline risk.")

        return factors[:6]

    @staticmethod
    def _guidance_for_level(level: str) -> tuple[str, str]:
        if level == "high":
            return (
                "High-priority case. Please review vitals and symptom context before consultation.",
                "High risk category detected. Seek timely medical consultation. This is not a diagnosis.",
            )
        if level == "medium":
            return (
                "Moderate-risk case. Monitor trends and verify risk factors during consultation.",
                "Medium risk category detected. Follow preventive care and consult a doctor as needed.",
            )
        return (
            "Lower-risk case based on available inputs. Continue routine monitoring.",
            "Low risk category detected. Maintain healthy habits and regular check-ups.",
        )

    def predict(self, features: Dict[str, Any]) -> RiskPredictionResponse:
        if not features:
            raise ValueError("patient_features cannot be empty")

        normalized_features = self._normalize_features(features)
        row = pd.DataFrame([normalized_features])

        expected_cols = getattr(self.model, "feature_names_in_", None)
        if expected_cols is not None:
            # Keep prediction robust when input JSON is partial.
            missing = [col for col in expected_cols if col not in row.columns]
            for col in missing:
                row[col] = self._default_value_for_feature(str(col))
            row = row[list(expected_cols)]

        model_class, model_high_prob = self._model_based_risk(row)
        rule_class = self._rule_based_risk_class(normalized_features)

        # Hybrid decision: model-first, with rule-based guardrails for healthcare safety.
        if rule_class == "High Risk":
            predicted_class = "High Risk"
            risk_probability = max(model_high_prob, 0.9)
        elif rule_class == "Medium Risk":
            if model_class == "High Risk":
                predicted_class = "High Risk"
                risk_probability = max(model_high_prob, 0.8)
            else:
                predicted_class = "Medium Risk"
                risk_probability = max(model_high_prob, 0.55)
        else:
            predicted_class = model_class
            risk_probability = model_high_prob

        if predicted_class == "High Risk":
            confidence_breakdown = {
                "Low Risk": float(np.clip(1.0 - risk_probability, 0.0, 1.0)) * 0.2,
                "Medium Risk": float(np.clip(1.0 - risk_probability, 0.0, 1.0)) * 0.8,
                "High Risk": float(np.clip(risk_probability, 0.0, 1.0)),
            }
            risk_level = "high"
        elif predicted_class == "Medium Risk":
            medium_weight = float(np.clip(risk_probability, 0.0, 1.0))
            remaining = float(np.clip(1.0 - medium_weight, 0.0, 1.0))
            confidence_breakdown = {
                "Low Risk": remaining * 0.55,
                "Medium Risk": medium_weight,
                "High Risk": remaining * 0.45,
            }
            risk_level = "medium"
        else:
            confidence_breakdown = {
                "Low Risk": float(np.clip(1.0 - risk_probability, 0.0, 1.0)),
                "Medium Risk": float(np.clip(risk_probability, 0.0, 1.0)) * 0.3,
                "High Risk": float(np.clip(risk_probability, 0.0, 1.0)) * 0.7,
            }
            risk_level = "low"

        contributing_factors = self._explain_contributing_factors(normalized_features)
        doctor_note, patient_guidance = self._guidance_for_level(risk_level)

        return RiskPredictionResponse(
            predicted_class=str(predicted_class),
            risk_probability=float(risk_probability),
            confidence_breakdown=confidence_breakdown,
            risk_level=risk_level,
            contributing_factors=contributing_factors,
            doctor_note=doctor_note,
            patient_guidance=patient_guidance,
        )


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def default_features_json(risk_engine: RiskEngine) -> str:
    expected_cols = getattr(risk_engine.model, "feature_names_in_", None)
    if expected_cols is not None:
        default_features = {col: risk_engine._default_value_for_feature(str(col)) for col in expected_cols}
    else:
        default_features = {
            "Age": 45,
            "Gender": 1,
            "Symptoms": "none",
            "Symptom_Count": 1,
            "Glucose": 95,
            "BloodPressure": 120,
            "BMI": 24.5,
            "Age_Group": "Adult",
            "BMI_Category": "Normal",
            "BP_Category": "Normal",
        }
    return escape(json.dumps(default_features, indent=2))
