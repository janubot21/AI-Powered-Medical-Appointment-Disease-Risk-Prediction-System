import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    # Allow `python backend/api.py` to work by ensuring project root is on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.risk_engine import engine
from backend.services import shared


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
STATIC_DIR = PROJECT_ROOT / "frontend" / "static"
FAVICON_ICO_PATH = STATIC_DIR / "favicon.ico"
FAVICON_SVG_PATH = STATIC_DIR / "img" / "heartbeat.svg"

app = FastAPI(
    title="Healthcare Risk Prediction API",
    description="Live appointment-time risk estimation service with Logistic Regression plus real-world clinical threshold checks.",
    version="2.1.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RiskPredictionRequest(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Patient age in years.")
    gender: str = Field(..., description="Patient gender.")
    symptoms: list[str] | str = Field(..., description="Symptoms as a list or comma-separated string.")
    glucose: float = Field(..., ge=0, description="Glucose reading in mg/dL.")
    glucose_type: Literal["fasting", "random"] = Field(
        ...,
        description="Whether the glucose reading is fasting or random.",
    )
    systolic_bp: float = Field(..., ge=0, description="Systolic blood pressure in mmHg.")
    diastolic_bp: float = Field(..., ge=0, description="Diastolic blood pressure in mmHg.")
    bmi: float = Field(..., ge=0, description="Body Mass Index.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 47,
                    "gender": "Female",
                    "symptoms": ["fatigue", "shortness of breath", "cough", "dizziness"],
                    "glucose": 145.0,
                    "glucose_type": "fasting",
                    "systolic_bp": 148.0,
                    "diastolic_bp": 96.0,
                    "bmi": 31.2,
                },
                {
                    "age": 58,
                    "gender": "Male",
                    "symptoms": ["chest pain", "shortness of breath", "dizziness"],
                    "glucose": 52.0,
                    "glucose_type": "random",
                    "systolic_bp": 190.0,
                    "diastolic_bp": 124.0,
                    "bmi": 33.4,
                },
            ]
        }
    }


class RiskPredictionResponse(BaseModel):
    risk_group: str
    model_risk_group: str
    clinical_risk_group: str
    risk_probabilities: dict[str, float]
    booking_priority: str
    clinical_assessment: dict[str, Any]
    engineered_features: dict[str, Any]


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "healthcare-risk-prediction-api",
        "model": "trained_risk_stratification_model.pkl",
        "rules": "adult clinical glucose, blood pressure, and BMI thresholds enabled",
    }


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "message": "Healthcare Risk Prediction API is running.",
        "docs": "/docs",
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict-risk",
        "example_payload": {
            "age": 47,
            "gender": "Female",
            "symptoms": ["fatigue", "shortness of breath", "cough", "dizziness"],
            "glucose": 145.0,
            "glucose_type": "fasting",
            "systolic_bp": 148.0,
            "diastolic_bp": 96.0,
            "bmi": 31.2,
        },
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    if FAVICON_ICO_PATH.exists():
        return FileResponse(path=str(FAVICON_ICO_PATH))

    if FAVICON_SVG_PATH.exists():
        return RedirectResponse(url="/static/img/heartbeat.svg")

    return Response(status_code=204)


@app.get("/about", include_in_schema=False)
def about_page() -> FileResponse | Response:
    return shared.html_page(shared.HTML_PAGES["about"])


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": "Logistic Regression",
        "target": "Risk_Factor_Group",
        "classes": list(engine.model.classes_),
        "final_risk_benchmarks": ["Low", "Moderate", "High", "Critical"],
        "api_inputs": [
            "age",
            "gender",
            "symptoms",
            "glucose",
            "glucose_type",
            "systolic_bp",
            "diastolic_bp",
            "bmi",
        ],
        "notes": [
            "Blood pressure uses adult clinical systolic/diastolic thresholds.",
            "Glucose uses adult fasting or random glucose thresholds.",
            "The historical ML model was trained on legacy tabular features, so the engine maps live clinical inputs into the model's expected feature schema.",
        ],
    }


@app.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> RiskPredictionResponse:
    result = engine.predict(
        age=payload.age,
        gender=payload.gender,
        symptoms=payload.symptoms,
        glucose=payload.glucose,
        glucose_type=payload.glucose_type,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        bmi=payload.bmi,
    )
    return RiskPredictionResponse(**result)


if __name__ == "__main__":
    import uvicorn

    # If you want auto-reload, prefer:
    #   python -m uvicorn backend.api:app --reload --port 8000
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=False)
