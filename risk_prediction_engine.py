from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
FEATURE_DATASET_PATH = BASE_DIR / "Healthcare_FeatureEngineered.csv"


def default_value_for_feature(feature_name: str) -> Any:
    numeric_defaults = {
        "Age": 45,
        "Symptom_Count": 1,
        "Glucose": 95,
        "BloodPressure": 120,
        "BMI": 24.5,
    }
    categorical_defaults = {
        "Symptoms": "none",
        "Age_Group": "Adult",
        "BMI_Category": "Normal",
        "BP_Category": "Normal",
    }

    if feature_name in numeric_defaults:
        return numeric_defaults[feature_name]
    if feature_name == "Gender":
        return 1
    if feature_name in categorical_defaults:
        return categorical_defaults[feature_name]
    if feature_name.startswith("SYM_"):
        return 0
    return 0


def default_features_from_columns(columns: List[str]) -> Dict[str, Any]:
    return {col: default_value_for_feature(str(col)) for col in columns}


def build_patient_feature_lookup(
    dataset_path: Path,
    expected_feature_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    if not dataset_path.exists():
        return {}

    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return {}

    if "Patient_ID" not in df.columns:
        return {}

    available_cols = [col for col in expected_feature_cols if col in df.columns]
    lookup: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        raw_id = row["Patient_ID"]
        if pd.isna(raw_id):
            continue

        if isinstance(raw_id, float) and raw_id.is_integer():
            patient_id = str(int(raw_id))
        else:
            patient_id = str(raw_id).strip()

        features = default_features_from_columns(expected_feature_cols)
        for col in available_cols:
            value = row[col]
            features[col] = None if pd.isna(value) else value

        lookup[patient_id] = features

    return lookup


class RiskPredictionRequest(BaseModel):
    patient_features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary expected by the trained model.",
    )


class AppointmentBookingRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    doctor_id: str = Field(..., description="Assigned doctor identifier")
    appointment_time: datetime = Field(..., description="Requested appointment date-time")
    patient_features: Dict[str, Any] = Field(
        ...,
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
    def __init__(self, model_path: Path, label_encoder_path: Optional[Path] = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.label_encoder = None

        if label_encoder_path and label_encoder_path.exists():
            self.label_encoder = joblib.load(label_encoder_path)
        self.numeric_cols, self.categorical_cols = self._infer_column_groups()

    def _infer_column_groups(self) -> tuple[List[str], List[str]]:
        numeric_cols: List[str] = []
        categorical_cols: List[str] = []
        preprocessor = getattr(self.model, "named_steps", {}).get("preprocessor")
        if preprocessor is None or not hasattr(preprocessor, "transformers"):
            return numeric_cols, categorical_cols

        for name, _, cols in preprocessor.transformers:
            col_list = [str(col) for col in cols]
            if name == "num":
                numeric_cols.extend(col_list)
            elif name == "cat":
                categorical_cols.extend(col_list)

        return numeric_cols, categorical_cols

    def _coerce_gender(self, value: Any) -> float:
        if value is None:
            return np.nan
        if isinstance(value, (int, float)):
            return float(value)

        normalized = str(value).strip().lower()
        if normalized in {"male", "m"}:
            return 1.0
        if normalized in {"female", "f"}:
            return 0.0

        try:
            return float(normalized)
        except ValueError as exc:
            raise ValueError("Gender must be numeric (e.g. 0/1) or Male/Female.") from exc

    def _coerce_features(self, row: pd.DataFrame) -> pd.DataFrame:
        converted: Dict[str, Any] = {}

        for col in row.columns:
            value = row.iloc[0][col]

            if col in self.numeric_cols:
                if value in (None, ""):
                    converted[col] = np.nan
                elif col == "Gender":
                    converted[col] = self._coerce_gender(value)
                else:
                    try:
                        converted[col] = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Feature '{col}' must be numeric.") from exc
                continue

            if col in self.categorical_cols:
                converted[col] = np.nan if value in (None, "") else str(value)
                continue

            converted[col] = value

        return pd.DataFrame([converted], columns=list(row.columns))

    def _risk_level(self, probability: float) -> str:
        if probability >= 0.75:
            return "high"
        if probability >= 0.4:
            return "medium"
        return "low"

    def predict(self, features: Dict[str, Any]) -> RiskPredictionResponse:
        if not features:
            raise ValueError("patient_features cannot be empty")

        row = pd.DataFrame([features])

        # Align with model training schema when available.
        expected_cols = getattr(self.model, "feature_names_in_", None)
        if expected_cols is not None:
            missing = [col for col in expected_cols if col not in row.columns]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            row = row[list(expected_cols)]
            row = self._coerce_features(row)

        predicted_idx = self.model.predict(row)[0]

        if self.label_encoder is not None:
            predicted_class = self.label_encoder.inverse_transform([predicted_idx])[0]
        else:
            predicted_class = str(predicted_idx)

        confidence_breakdown: Dict[str, float] = {}
        risk_probability = 0.0

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(row)[0]
            classes = getattr(self.model, "classes_", np.arange(len(probabilities)))

            if self.label_encoder is not None:
                class_names = self.label_encoder.inverse_transform(classes)
            else:
                class_names = [str(c) for c in classes]

            confidence_breakdown = {
                str(name): float(prob) for name, prob in zip(class_names, probabilities)
            }
            risk_probability = float(np.max(probabilities))
        else:
            # Fallback if model has no probability API.
            confidence_breakdown = {predicted_class: 1.0}
            risk_probability = 1.0

        return RiskPredictionResponse(
            predicted_class=str(predicted_class),
            risk_probability=risk_probability,
            confidence_breakdown=confidence_breakdown,
            risk_level=self._risk_level(risk_probability),
        )


app = FastAPI(
    title="Disease Risk Prediction Engine",
    version="1.0.0",
    description="Live disease risk estimation integrated with appointment booking.",
)


try:
    risk_engine = RiskEngine(MODEL_PATH, LABEL_ENCODER_PATH)
except Exception as exc:  # pragma: no cover - startup guard
    raise RuntimeError(f"Failed to initialize risk engine: {exc}") from exc

EXPECTED_FEATURE_COLUMNS = [
    str(col) for col in getattr(risk_engine.model, "feature_names_in_", [])
]
DEFAULT_FEATURES_TEMPLATE = default_features_from_columns(EXPECTED_FEATURE_COLUMNS)
PATIENT_FEATURE_LOOKUP = build_patient_feature_lookup(
    FEATURE_DATASET_PATH,
    EXPECTED_FEATURE_COLUMNS,
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    default_features_json = escape(json.dumps(DEFAULT_FEATURES_TEMPLATE, indent=2))
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Disease Risk Prediction Engine</title>
  <style>
    :root {
      --ink: #1b1f2a;
      --cloud: #f6f7fb;
      --teal: #0b7a75;
      --sun: #ff9f1c;
      --mint: #2ec4b6;
      --paper: #ffffff;
      --line: #d9dce7;
      --danger: #9f1239;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "Consolas", "Lucida Console", monospace;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 18%, rgba(46, 196, 182, 0.15), transparent 42%),
        radial-gradient(circle at 88% 8%, rgba(255, 159, 28, 0.18), transparent 34%),
        linear-gradient(150deg, #f1f5ff, #fcfcff 48%, #f8fbff);
      min-height: 100vh;
    }

    .shell {
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 18px 36px;
      animation: rise 500ms ease-out;
    }

    h1, h2 {
      font-family: "Palatino Linotype", "Book Antiqua", serif;
      margin: 0;
      letter-spacing: 0.02em;
    }

    h1 {
      font-size: clamp(1.7rem, 3.5vw, 2.5rem);
    }

    .subhead {
      margin: 10px 0 22px;
      color: #394058;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(29, 39, 71, 0.08);
    }

    label {
      display: block;
      margin: 10px 0 6px;
      font-weight: 600;
      font-size: 0.95rem;
    }

    input, textarea, button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      font-family: inherit;
      font-size: 0.95rem;
    }

    input, textarea {
      padding: 10px 12px;
      background: #fbfcff;
    }

    textarea {
      min-height: 120px;
      resize: vertical;
    }

    button {
      margin-top: 14px;
      padding: 10px 12px;
      border: 0;
      background: linear-gradient(120deg, var(--teal), #1164a3);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
      transition: transform 160ms ease, filter 160ms ease;
    }

    button:hover {
      transform: translateY(-1px);
      filter: brightness(1.05);
    }

    .output {
      margin-top: 14px;
      padding: 12px;
      border-radius: 10px;
      background: #f7f9ff;
      border: 1px solid #d5dcf0;
      white-space: pre-wrap;
      overflow-x: auto;
      min-height: 58px;
    }

    .quicklinks {
      margin-top: 14px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .quicklinks a {
      color: var(--teal);
      text-decoration: none;
      font-weight: 700;
    }

    .error {
      color: var(--danger);
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <h1>Disease Risk Prediction Engine</h1>
    <p class="subhead">Interactive frontend for risk scoring and appointment booking.</p>

    <section class="grid">
      <article class="card">
        <h2>Predict Risk</h2>
        <label for="riskFeatures">Patient Features (JSON)</label>
        <textarea id="riskFeatures">__DEFAULT_FEATURES_JSON__</textarea>
        <button id="predictBtn" type="button">Run Prediction</button>
        <pre class="output" id="predictOutput">Waiting for input...</pre>
      </article>

      <article class="card">
        <h2>Book Appointment</h2>
        <label for="patientId">Patient ID</label>
        <input id="patientId" value="1" placeholder="e.g. 1" />
        <label for="doctorId">Doctor ID</label>
        <input id="doctorId" value="2" placeholder="D-209" />
        <label for="appointmentTime">Appointment Time</label>
        <input id="appointmentTime" type="datetime-local" />
        <label for="bookFeatures">Patient Features (JSON)</label>
        <textarea id="bookFeatures">__DEFAULT_FEATURES_JSON__</textarea>
        <button id="bookBtn" type="button">Book Appointment</button>
        <pre class="output" id="bookOutput">Waiting for input...</pre>
      </article>
    </section>

    <div class="quicklinks">
      <a href="/docs" target="_blank" rel="noreferrer">Open Swagger Docs</a>
      <a href="/health" target="_blank" rel="noreferrer">Check Health</a>
    </div>
  </main>

  <script>
    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data && data.detail ? data.detail : "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    }

    function parseFeatures(textareaId) {
      const raw = document.getElementById(textareaId).value;
      try {
        return JSON.parse(raw);
      } catch (err) {
        throw new Error("Invalid JSON in patient features.");
      }
    }

    function renderOutput(targetId, value, isError) {
      const el = document.getElementById(targetId);
      el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
      el.className = isError ? "output error" : "output";
    }

    function ensureDefaultAppointmentTime() {
      const input = document.getElementById("appointmentTime");
      if (input.value) {
        return;
      }

      const dt = new Date();
      dt.setMinutes(dt.getMinutes() + 30);
      dt.setSeconds(0, 0);
      const localIso = new Date(dt.getTime() - dt.getTimezoneOffset() * 60000)
        .toISOString()
        .slice(0, 16);
      input.value = localIso;
    }

    async function loadFeaturesByPatientId() {
      const patientId = document.getElementById("patientId").value.trim();
      if (!patientId) {
        return;
      }

      try {
        const res = await fetch(`/patient-features/${encodeURIComponent(patientId)}`);
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data && data.detail ? data.detail : "Unable to load patient features.");
        }

        const pretty = JSON.stringify(data.patient_features, null, 2);
        document.getElementById("riskFeatures").value = pretty;
        document.getElementById("bookFeatures").value = pretty;

        const message = data.found
          ? `Loaded dataset features for patient_id=${data.patient_id}`
          : `No dataset row found for patient_id=${data.patient_id}. Loaded default template.`;
        renderOutput("bookOutput", message, false);
      } catch (err) {
        renderOutput("bookOutput", err.message, true);
      }
    }

    document.getElementById("predictBtn").addEventListener("click", async () => {
      try {
        const patient_features = parseFeatures("riskFeatures");
        renderOutput("predictOutput", "Submitting...", false);
        const data = await postJson("/predict-risk", { patient_features });
        renderOutput("predictOutput", data, false);
      } catch (err) {
        renderOutput("predictOutput", err.message, true);
      }
    });

    document.getElementById("bookBtn").addEventListener("click", async () => {
      try {
        const patient_features = parseFeatures("bookFeatures");
        const appointmentTime = document.getElementById("appointmentTime").value;
        if (!appointmentTime) {
          throw new Error("Please select a valid appointment time.");
        }
        const payload = {
          patient_id: document.getElementById("patientId").value.trim(),
          doctor_id: document.getElementById("doctorId").value.trim(),
          appointment_time: appointmentTime,
          patient_features
        };
        renderOutput("bookOutput", "Submitting...", false);
        const data = await postJson("/book-appointment", payload);
        renderOutput("bookOutput", data, false);
      } catch (err) {
        renderOutput("bookOutput", err.message, true);
      }
    });

    document.getElementById("patientId").addEventListener("change", loadFeaturesByPatientId);
    document.getElementById("patientId").addEventListener("blur", loadFeaturesByPatientId);
    window.addEventListener("load", () => {
      ensureDefaultAppointmentTime();
      loadFeaturesByPatientId();
    });
  </script>
</body>
</html>
"""
    return html_template.replace("__DEFAULT_FEATURES_JSON__", default_features_json)


@app.get("/patient-features/{patient_id}")
def get_patient_features(patient_id: str) -> Dict[str, Any]:
    normalized = str(patient_id).strip()
    features = PATIENT_FEATURE_LOOKUP.get(normalized)
    if features is not None:
        return {"patient_id": normalized, "found": True, "patient_features": features}

    return {
        "patient_id": normalized,
        "found": False,
        "patient_features": DEFAULT_FEATURES_TEMPLATE,
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "risk_prediction_engine"}


@app.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> RiskPredictionResponse:
    try:
        return risk_engine.predict(payload.patient_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/book-appointment", response_model=AppointmentBookingResponse)
def book_appointment(payload: AppointmentBookingRequest) -> AppointmentBookingResponse:
    try:
        risk_assessment = risk_engine.predict(payload.patient_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {exc}") from exc

    # Replace this with actual DB persistence / scheduling logic.
    booking_status = "confirmed"

    return AppointmentBookingResponse(
        booking_status=booking_status,
        patient_id=payload.patient_id,
        doctor_id=payload.doctor_id,
        appointment_time=payload.appointment_time,
        risk_assessment=risk_assessment,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("risk_prediction_engine:app", host="0.0.0.0", port=8000, reload=True)
