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
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "decision_tree_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
FEATURE_DATA_PATH = BASE_DIR / "Healthcare_FeatureEngineered.csv"


class RiskPredictionRequest(BaseModel):
    patient_features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary expected by the trained model.",
    )


class AppointmentBookingRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    doctor_id: str = Field(..., description="Assigned doctor identifier")
    appointment_time: datetime = Field(..., description="Requested appointment date-time")
    patient_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Feature dictionary expected by the trained model.",
    )


class RiskPredictionResponse(BaseModel):
    predicted_class: str
    risk_probability: float
    confidence_breakdown: Dict[str, float]
    risk_level: str


class AppointmentBookingResponse(BaseModel):
    booking_status: str
    patient_id: str
    doctor_id: str
    appointment_time: datetime
    risk_assessment: RiskPredictionResponse


class RiskEngine:
    HIGH_RISK_SYMPTOMS = {"chest_pain", "diarrhea", "diarrhoea", "insomnia", "dizziness"}
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

    def __init__(self, model_path: Path, label_encoder_path: Optional[Path] = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.label_encoder = None

        if label_encoder_path and label_encoder_path.exists():
            self.label_encoder = joblib.load(label_encoder_path)
        self.patient_feature_map = self._load_patient_feature_map(FEATURE_DATA_PATH)

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

        missing_expected = [c for c in expected_cols if c not in df.columns]
        if missing_expected:
            return {}

        feature_map: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            pid = self._normalize_patient_id(row["Patient_ID"])
            features = {}
            for col in expected_cols:
                val = row[col]
                features[col] = None if pd.isna(val) else val
            feature_map[pid] = features
        return feature_map

    def get_patient_features(self, patient_id: str) -> Dict[str, Any]:
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

    def _extract_reported_symptoms(self, features: Dict[str, Any]) -> set[str]:
        found: set[str] = set()

        raw_symptoms = features.get("Symptoms")
        if isinstance(raw_symptoms, str):
            parts = [p.strip() for p in re.split(r"[,;|]+", raw_symptoms) if p.strip()]
            found.update(self._normalize_symptom_name(p) for p in parts)

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

        if (age > 50 and symptom_count > 0) or glucose > 150 or blood_pressure > 120:
            return "High Risk"
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
        # Model-first scoring; maps model output into a binary High/Low risk view.
        prediction = self.model.predict(row)[0]

        decoded_label = str(prediction)
        if self.label_encoder is not None:
            try:
                decoded_label = str(self.label_encoder.inverse_transform([int(prediction)])[0])
            except Exception:
                decoded_label = str(prediction)

        decoded_lower = decoded_label.strip().lower()
        model_class = "High Risk" if "high" in decoded_lower else "Low Risk"

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
                if high_idx is not None and 0 <= high_idx < len(probs):
                    high_prob = float(probs[high_idx])
                else:
                    high_prob = 1.0 if model_class == "High Risk" else 0.0
            except Exception:
                high_prob = 1.0 if model_class == "High Risk" else 0.0

        return model_class, float(np.clip(high_prob, 0.0, 1.0))

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

        # Hybrid decision: model-first, with rule-based high-risk override as guardrail.
        if rule_class == "High Risk":
            predicted_class = "High Risk"
            risk_probability = max(model_high_prob, 0.9)
        else:
            predicted_class = model_class
            risk_probability = model_high_prob

        confidence_breakdown = {
            "Low Risk": float(np.clip(1.0 - risk_probability, 0.0, 1.0)),
            "High Risk": float(np.clip(risk_probability, 0.0, 1.0)),
        }
        risk_level = "high" if predicted_class == "High Risk" else "low"

        return RiskPredictionResponse(
            predicted_class=str(predicted_class),
            risk_probability=float(risk_probability),
            confidence_breakdown=confidence_breakdown,
            risk_level=risk_level,
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
