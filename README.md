# AI-Powered Medical Appointment and Disease Risk Prediction System

This project is a multi-role healthcare portal that combines disease-risk prediction, appointment booking, and role-based workflows for patients, doctors, and nurses.

## What It Includes

- FastAPI API layer for prediction and booking endpoints
- Flask portal for patient, doctor, and nurse dashboards
- Machine-learning risk prediction using saved scikit-learn models
- Patient profile storage and appointment persistence with SQLite
- CSV-backed account and dataset bootstrap flow
- Optional OpenAI-powered patient chat support
- PDF report generation for project documentation

## Project Structure

```text
backend/         FastAPI, Flask portal, auth, prediction, persistence
backend/routes/  Split Flask route registrations for common, patient, and staff flows
frontend/        Jinja templates and static assets
documentation/   Report generator and generated report assets
tests/           Smoke tests for auth, config, persistence, and booking
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a local environment file:

```powershell
Copy-Item .env.example .env
```

4. Update `.env` with your local values, especially `FLASK_SECRET_KEY` and any optional OpenAI settings.
5. Initialize local dataset/account files:

```powershell
cd backend
python init_datasets.py
```

## Run The App

From the `backend/` directory:

```powershell
uvicorn main:app --reload
```

Useful routes:

- `http://127.0.0.1:8000/portal/login`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

Protected API routes now require a bearer token returned from:

- `POST /patient-login`
- `POST /doctor-login`
- `POST /nurse-login`

## Tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

## Environment Variables

- `FLASK_SECRET_KEY`: required for stable Flask sessions across restarts
- `COOKIE_SECURE`: set to `1` in HTTPS deployments
- `OPENAI_API_KEY`: optional, enables patient AI chat
- `OPENAI_CHAT_MODEL`: optional, defaults to `gpt-4o-mini`
- `*_CSV_PATH` and `PATIENT_DB_PATH`: optional dataset overrides

## Security Notes

- `.env.example` now contains placeholders only and is not loaded at runtime
- Patient, doctor, and nurse passwords are stored as PBKDF2 hashes
- Legacy plaintext patient passwords have been migrated out of `backend/datasets/patients.csv`
- Flask now uses a generated secret key when no explicit secret is configured, instead of a fixed insecure fallback
- Sensitive FastAPI medical endpoints now require signed bearer tokens and enforce role scope checks
- API and portal appointments now share the same SQLite-backed storage

## Current Limitations

- The ML outputs are educational/project-focused and should not be treated as clinical advice
- Some datasets in `backend/datasets/` are sample/project data rather than production-grade records

## Recommended Next Steps

- Split remaining Flask route groups into dedicated modules
- Add broader route and UI tests
- Add deployment config for a single-command local/prod startup
- Expand README screenshots and demo notes
