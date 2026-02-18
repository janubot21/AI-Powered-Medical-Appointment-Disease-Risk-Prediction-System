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
MODEL_PATH = (
    BASE_DIR / "best_model.pkl"
    if (BASE_DIR / "best_model.pkl").exists()
    else BASE_DIR / "decision_tree_model.pkl"
)
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
        symptoms = self._extract_reported_symptoms(features)
        if symptoms & self.HIGH_RISK_SYMPTOMS:
            return "High"
        if symptoms & self.MEDIUM_RISK_SYMPTOMS:
            return "Medium"
        return "Low"

    def predict(self, features: Dict[str, Any]) -> RiskPredictionResponse:
        if not features:
            raise ValueError("patient_features cannot be empty")

        normalized_features = self._normalize_features(features)
        row = pd.DataFrame([normalized_features])

        expected_cols = getattr(self.model, "feature_names_in_", None)
        if expected_cols is not None:
            missing = [col for col in expected_cols if col not in row.columns]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            row = row[list(expected_cols)]

        predicted_class = self._rule_based_risk_class(normalized_features)
        confidence_breakdown = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
        confidence_breakdown[predicted_class] = 1.0
        risk_probability = 1.0

        return RiskPredictionResponse(
            predicted_class=str(predicted_class),
            risk_probability=risk_probability,
            confidence_breakdown=confidence_breakdown,
            risk_level=predicted_class.lower(),
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
    def default_value_for_feature(feature_name: str) -> Any:
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

    expected_cols = getattr(risk_engine.model, "feature_names_in_", None)
    if expected_cols is not None:
        default_features = {col: default_value_for_feature(str(col)) for col in expected_cols}
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
