from __future__ import annotations

from typing import Any, Dict

from flask import jsonify, redirect, render_template, request, session, url_for


def register_patient_routes(app: Any) -> None:
    import flask_app as fa

    @app.route("/patient/signup", methods=["GET", "POST"])
    def patient_signup() -> Any:
        errors: list[str] = []
        form_data = {"name": ""}

        if request.method == "POST":
            name = request.form.get("name", "").strip()
            password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()
            form_data["name"] = name

            if not name:
                errors.append("Patient Name is required.")
            if not fa.PASSWORD_POLICY_PATTERN.fullmatch(password):
                errors.append(fa.PASSWORD_POLICY_MESSAGE)
            if password != confirm_password:
                errors.append("Password and Confirm Password must match.")

            if not errors:
                try:
                    patient_id = fa._create_patient_account(name, "", password)
                    fa._clear_doctor_session()
                    fa._clear_nurse_session()
                    session["patient_id"] = patient_id
                    session["patient_name"] = name
                    session["patient_authenticated"] = False
                    session["role"] = "patient"
                    session["allow_health_details"] = True
                    return redirect(url_for("health_details", patient_id=patient_id))
                except ValueError as exc:
                    errors.append(str(exc))
                except Exception as exc:  # pragma: no cover - guard
                    fa.logger.exception("Patient portal signup failed for name=%s", name)
                    errors.append(f"Signup failed: {exc}")

        return render_template("flask_patient_signup.html", errors=errors, form_data=form_data)

    @app.route("/patient/health-details", methods=["GET", "POST"])
    def health_details() -> Any:
        errors: list[str] = []
        if not session.get("allow_health_details"):
            patient_id = session.get("patient_id")
            if patient_id:
                return redirect(url_for("book_appointment"))
            return redirect(url_for("patient_signup"))

        patient_id = (
            session.get("patient_id", "")
            or request.args.get("patient_id", "").strip()
            or request.form.get("patient_id", "").strip()
        )
        form_data = {
            "patient_id": patient_id,
            "age": "",
            "gender": "",
            "symptoms": "",
            "symptom_count": "",
            "glucose": "",
            "blood_pressure": "",
            "height_cm": "",
            "weight_kg": "",
            "bmi": "",
            "smoking_habit": "",
            "alcohol_habit": "",
            "average_sleep_hours": "",
            "medical_history": "",
            "family_history": "",
        }

        if request.method == "POST":
            form_data.update(
                {
                    "age": request.form.get("age", "").strip(),
                    "gender": request.form.get("gender", "").strip().lower(),
                    "symptoms": fa._sanitize_symptoms_text(request.form.get("symptoms", "")),
                    "symptom_count": request.form.get("symptom_count", "").strip(),
                    "glucose": request.form.get("glucose", "").strip(),
                    "blood_pressure": request.form.get("blood_pressure", "").strip(),
                    "height_cm": request.form.get("height_cm", "").strip(),
                    "weight_kg": request.form.get("weight_kg", "").strip(),
                    "bmi": request.form.get("bmi", "").strip(),
                    "smoking_habit": request.form.get("smoking_habit", "").strip().lower(),
                    "alcohol_habit": request.form.get("alcohol_habit", "").strip().lower(),
                    "average_sleep_hours": request.form.get("average_sleep_hours", "").strip(),
                    "medical_history": request.form.get("medical_history", "").strip(),
                    "family_history": request.form.get("family_history", "").strip().lower(),
                }
            )

            if not patient_id:
                errors.append("Patient ID is missing. Please signup again.")
            if form_data["gender"] not in {"male", "female", "other"}:
                errors.append("Gender must be one of: male, female, other.")
            if form_data["smoking_habit"] not in {"yes", "no"}:
                errors.append("Smoking Habit must be yes or no.")
            if form_data["alcohol_habit"] not in {"yes", "no"}:
                errors.append("Alcohol Habit must be yes or no.")
            if form_data["family_history"] not in {"yes", "no"}:
                errors.append("Family History must be yes or no.")
            if not form_data["symptoms"]:
                errors.append("Symptoms are required.")

            try:
                age = fa._to_int(form_data["age"], "Age", 0, 130)
                symptom_count = fa._to_int(form_data["symptom_count"], "Symptom_Count", 0, 100)
                glucose = fa._to_float(form_data["glucose"], "Glucose", 0, 1000)
                blood_pressure = fa._to_float(form_data["blood_pressure"], "BloodPressure", 0, 400)
                height_cm = fa._to_float(form_data["height_cm"], "Height (cm)", 1, 300)
                weight_kg = fa._to_float(form_data["weight_kg"], "Weight (kg)", 1, 500)
                average_sleep_hours = fa._normalize_average_sleep_hours(form_data["average_sleep_hours"], required=True)
                bmi = fa._calculate_bmi(height_cm, weight_kg)
                form_data["bmi"] = f"{bmi:.2f}"
            except ValueError as exc:
                errors.append(str(exc))
                age = symptom_count = 0
                glucose = blood_pressure = bmi = average_sleep_hours = 0.0
                height_cm = weight_kg = 0.0

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
                    "Height_cm": height_cm,
                    "Weight_kg": weight_kg,
                    "Smoking_Habit": form_data["smoking_habit"],
                    "Alcohol_Habit": form_data["alcohol_habit"],
                    "Average_Sleep_Hours": average_sleep_hours,
                    "Medical_History": form_data["medical_history"],
                    "Family_History": form_data["family_history"],
                    fa.HEALTH_DATA_SUBMITTED_AT_COLUMN: fa._utc_now_iso(),
                }
                try:
                    fa.append_to_new_patient_csv(row)
                    fa._mark_health_details_submitted(patient_id)
                except Exception as exc:  # pragma: no cover - guard
                    fa.logger.exception("Could not save health details for patient_id=%s", patient_id)
                    errors.append(f"Could not save health details: {exc}")

                if not errors:
                    session["health_confirmation"] = row
                    session.pop("patient_id", None)
                    session.pop("patient_name", None)
                    session.pop("allow_health_details", None)
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
        form_data = {"patient_name": "", "id_type": "", "id_number": "", "login_using": "patient_name"}

        if request.method == "GET":
            fa._clear_patient_session()
            if session.get("role") == "patient":
                session.pop("role", None)

        if request.method == "POST":
            login_using = request.form.get("login_using", "patient_name").strip().lower()
            if login_using not in {"patient_name", "id_number"}:
                login_using = "patient_name"
            patient_name = request.form.get("patient_name", "").strip()
            id_type = request.form.get("id_type", "").strip().lower()
            id_number = request.form.get("id_number", "").strip().upper()
            password = request.form.get("password", "").strip()
            form_data["patient_name"] = patient_name
            form_data["id_type"] = id_type
            form_data["id_number"] = id_number
            form_data["login_using"] = login_using

            unique_code = ""
            if login_using == "patient_name":
                if not patient_name:
                    errors.append("Enter Patient Name.")
            else:
                if not id_number:
                    errors.append("Enter ID Number.")
                else:
                    try:
                        unique_code = fa._compose_unique_code(id_type, id_number)
                    except ValueError as exc:
                        errors.append(str(exc))
            if not password:
                errors.append("Password is required.")

            if not errors:
                try:
                    patient = fa._authenticate_patient(patient_name, unique_code, password)
                    if not patient:
                        raise ValueError("Invalid patient name, unique code, or password.")
                    patient_id = str(patient.get("patient_id", "")).strip()
                    resolved_name = str(patient.get("name", "")).strip() or patient_name
                    fa._clear_doctor_session()
                    fa._clear_nurse_session()
                    session["patient_id"] = patient_id
                    session["patient_name"] = resolved_name
                    session["patient_authenticated"] = True
                    session["role"] = "patient"
                    session.pop("health_confirmation", None)
                    session.pop("allow_health_details", None)
                    return render_template(
                        "flask_patient_login.html",
                        errors=[],
                        form_data={"patient_name": "", "id_type": "", "id_number": "", "login_using": "patient_name"},
                        login_success=True,
                        redirect_url=url_for("book_appointment"),
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                except Exception as exc:  # pragma: no cover - guard
                    fa.logger.exception("Patient portal login failed for patient_name=%s", patient_name)
                    errors.append(f"Login failed: {exc}")

        return render_template("flask_patient_login.html", errors=errors, form_data=form_data)

    @app.route("/patient/book-appointment")
    @fa.patient_required()
    def book_appointment() -> Any:
        patient_id = session.get("patient_id")
        patient_name = str(session.get("patient_name", "")).strip()
        if not patient_name:
            patient_name = fa._get_patient_name_by_id(str(patient_id).strip())
        doctor_ids = fa._load_doctor_ids()
        patient_features: Dict[str, Any] = {}
        try:
            patient_features = fa.get_features_for_patient(str(patient_id).strip())
        except ValueError:
            patient_features = {}
        return render_template(
            "flask_book_appointment.html",
            patient_id=patient_id,
            doctor_ids=doctor_ids,
            patient_name=patient_name,
            patient_features=patient_features,
        )

    @app.route("/patient/ai-chat")
    @fa.patient_required()
    def patient_ai_chat() -> Any:
        patient_id = str(session.get("patient_id", "")).strip()
        patient_name = str(session.get("patient_name", "")).strip() or fa._get_patient_name_by_id(patient_id) or "Patient"
        initial_payload = fa._patient_chat_response(patient_id, "Give my patient summary and precautions.")
        return render_template(
            "flask_patient_ai_chat.html",
            patient_id=patient_id,
            patient_name=patient_name,
            initial_reply=initial_payload.get("reply", ""),
            initial_snapshot=initial_payload.get("patient_snapshot", {}),
            initial_precautions=initial_payload.get("precautions", []),
            initial_suggested_doctors=initial_payload.get("suggested_doctors", []),
        )

    @app.post("/patient/ai-chat/message")
    @fa.patient_required(api=True)
    def patient_ai_chat_message() -> Any:
        patient_id = str(session.get("patient_id", "")).strip()
        payload = request.get_json(silent=True) or {}
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"detail": "message is required"}), 400
        return jsonify(fa._patient_chat_response(patient_id, message))

    @app.route("/patient/logout")
    def patient_logout() -> Any:
        session.pop("patient_id", None)
        session.pop("patient_name", None)
        session.pop("patient_authenticated", None)
        if session.get("role") == "patient":
            session.pop("role", None)
        session.pop("health_confirmation", None)
        session.pop("allow_health_details", None)
        return redirect(url_for("role_login"))

    @app.route("/patient/booking-confirmation")
    @fa.patient_required()
    def patient_booking_confirmation() -> Any:
        patient_id = str(session.get("patient_id", "")).strip()
        patient_name = str(session.get("patient_name", "")).strip() or fa._get_patient_name_by_id(patient_id)
        normalized_pid = fa._normalize_patient_id(patient_id)

        patient_appointments: list[Dict[str, Any]] = []
        stored_appointments = fa.patient_db.list_appointments(patient_id=patient_id)
        for item in stored_appointments:
            if fa._normalize_patient_id(item.get("patient_id")) != normalized_pid:
                continue
            dt = fa._parse_appointment_time(item.get("appointment_time"))
            doctor_profile = fa._doctor_profile_by_id(str(item.get("doctor_id", "")).strip())
            patient_appointments.append(
                {
                    "appointment_id": str(item.get("appointment_id", "")),
                    "patient_name": str(item.get("patient_name", "")).strip() or patient_name,
                    "doctor_id": str(item.get("doctor_id", "")).strip(),
                    "doctor_name": doctor_profile.get("doctor_name", "").strip() or str(item.get("doctor_id", "")).strip(),
                    "appointment_type": str(item.get("appointment_type", "")).strip(),
                    "appointment_time": dt,
                    "date_label": fa._format_date_label(dt),
                    "time_label": fa._format_time_label(dt),
                    "time_value": fa._format_time_value(dt),
                    "history_status": fa._appointment_history_status(dt),
                    "appointment_priority": str(item.get("appointment_priority", "")).strip() or "Normal",
                    "recommended_slot": str(item.get("recommended_slot", "")).strip() or "Next Available Date",
                    "priority_badge_text": str(item.get("priority_badge_text", "")).strip() or "Normal Appointment",
                    "priority_badge_color": str(item.get("priority_badge_color", "")).strip() or "green",
                }
            )

        patient_appointments.sort(key=lambda x: x.get("appointment_time") or fa.datetime.min, reverse=True)

        now = fa.datetime.now()
        upcoming = [a for a in patient_appointments if fa._is_upcoming_appointment_date(a.get("appointment_time"), reference=now)]
        past = [a for a in patient_appointments if not fa._is_upcoming_appointment_date(a.get("appointment_time"), reference=now)]
        upcoming.sort(key=lambda x: x.get("appointment_time") or fa.datetime.max)

        patient_features: Dict[str, Any] = {}
        if patient_appointments:
            latest_source = next(
                (
                    ap
                    for ap in stored_appointments
                    if fa._normalize_patient_id(ap.get("patient_id")) == normalized_pid and isinstance(ap.get("patient_features"), dict)
                ),
                None,
            )
            if latest_source:
                patient_features = dict(latest_source.get("patient_features") or {})
        if not patient_features:
            try:
                patient_features = fa.get_features_for_patient(patient_id)
            except ValueError:
                patient_features = {}

        return render_template(
            "flask_booking_confirmation.html",
            patient_id=patient_id,
            patient_name=patient_name,
            patient_features=patient_features,
            upcoming_appointments=upcoming,
            past_appointments=past,
        )

    @app.post("/patient/appointments/cancel")
    @fa.patient_required(api=True)
    def patient_cancel_appointment() -> Any:
        patient_id = str(session.get("patient_id", "")).strip()
        appointment_id = str(request.form.get("appointment_id", "")).strip()
        if not appointment_id:
            return redirect(url_for("patient_booking_confirmation"))

        fa.patient_db.delete_appointment(appointment_id, patient_id=patient_id)
        return redirect(url_for("patient_booking_confirmation"))

    @app.get("/patient/features/<patient_id>")
    def patient_features(patient_id: str) -> Any:
        if fa._is_patient_authenticated():
            session_pid = fa._normalize_patient_id(session.get("patient_id"))
            requested_pid = fa._normalize_patient_id(patient_id)
            if session_pid != requested_pid:
                return jsonify({"detail": "Forbidden: you can only access your own patient record."}), 403
        elif not fa._is_doctor_authenticated():
            return jsonify({"detail": "Unauthorized"}), 401

        try:
            features = fa.get_features_for_patient(patient_id)
            return jsonify({"patient_id": patient_id, "patient_features": features})
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 404
        except Exception as exc:  # pragma: no cover - guard
            fa.logger.exception("Patient feature lookup failed in portal for patient_id=%s", patient_id)
            return jsonify({"detail": f"Patient lookup failed: {exc}"}), 500

    @app.post("/patient/features/<patient_id>")
    @fa.patient_required(api=True)
    def update_patient_features(patient_id: str) -> Any:
        session_pid = fa._normalize_patient_id(session.get("patient_id"))
        requested_pid = fa._normalize_patient_id(patient_id)
        if not session_pid or requested_pid != session_pid:
            return jsonify({"detail": "Forbidden: you can only update your own patient record."}), 403

        payload = request.get_json(silent=True) or {}
        patient_features = payload.get("patient_features")
        if not isinstance(patient_features, dict) or not patient_features:
            return jsonify({"detail": "patient_features is required"}), 400

        try:
            fa.upsert_new_patient_csv_from_features(patient_id, patient_features)
            refreshed_features = fa.get_features_for_patient(patient_id)
            return jsonify({"patient_id": patient_id, "patient_features": refreshed_features})
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - guard
            fa.logger.exception("Could not update patient features in portal for patient_id=%s", patient_id)
            return jsonify({"detail": f"Could not update patient features: {exc}"}), 500

    @app.post("/patient/book-appointment-submit")
    @fa.patient_required(api=True)
    def submit_appointment() -> Any:
        payload = request.get_json(silent=True) or {}
        patient_id = str(payload.get("patient_id", "")).strip()
        patient_name = (
            str(payload.get("patient_name", "")).strip()
            or str(session.get("patient_name", "")).strip()
            or fa._get_patient_name_by_id(patient_id)
        )
        contact_info = str(payload.get("contact_info", "")).strip()
        doctor_id = str(payload.get("doctor_id", "")).strip()
        appointment_type = str(payload.get("appointment_type", "")).strip()
        appointment_time = str(payload.get("appointment_time", "")).strip()
        patient_features = payload.get("patient_features")

        if not patient_id:
            return jsonify({"detail": "patient_id is required"}), 400
        if not patient_name:
            return jsonify({"detail": "patient_name is required"}), 400
        if not contact_info:
            return jsonify({"detail": "contact_info is required"}), 400
        if not doctor_id:
            return jsonify({"detail": "doctor_id is required"}), 400
        if not appointment_type:
            return jsonify({"detail": "appointment_type is required"}), 400
        if not appointment_time:
            return jsonify({"detail": "appointment_time is required"}), 400

        session_pid = fa._normalize_patient_id(session.get("patient_id"))
        payload_pid = fa._normalize_patient_id(patient_id)
        if not session_pid or payload_pid != session_pid:
            return jsonify({"detail": "Forbidden: patient_id does not match your logged-in session."}), 403

        try:
            normalized_contact = fa._normalize_contact_info(contact_info)
            parsed_time = fa._ensure_future_appointment_time(fa.datetime.fromisoformat(appointment_time))
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400

        if patient_features is None:
            try:
                patient_features = fa.get_features_for_patient(patient_id)
            except ValueError as exc:
                return jsonify({"detail": str(exc)}), 400

        try:
            fa.upsert_new_patient_csv_from_features(patient_id, patient_features)
        except Exception as exc:  # pragma: no cover - guard
            fa.logger.exception("Could not update patient CSV before booking for patient_id=%s", patient_id)
            return jsonify({"detail": f"Could not update patient CSV: {exc}"}), 500

        try:
            risk_assessment = fa.risk_engine.predict(patient_features)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - guard
            fa.logger.exception("Risk assessment failed during patient booking for patient_id=%s", patient_id)
            return jsonify({"detail": f"Risk assessment failed: {exc}"}), 500

        risk_level_text = str(risk_assessment.risk_level or "").strip().lower()
        selected_doctor_profile = fa._doctor_profile_by_id(doctor_id)

        if fa._is_night_emergency_window(parsed_time) and not fa._is_emergency_doctor_profile(selected_doctor_profile):
            emergency_doctors = fa._available_emergency_doctors(parsed_time, exclude_doctor_id=doctor_id)
            alternative = emergency_doctors[0] if emergency_doctors else None
            return jsonify(
                {
                    "booking_status": "doctor_unavailable",
                    "reason": "non_emergency_doctor_night_window",
                    "risk_level": risk_level_text.title() or "Low",
                    "message": "Only emergency doctors are available during night hours (7:00 PM to 7:00 AM). Please choose an emergency doctor.",
                    "selected_doctor": selected_doctor_profile,
                    "alternative_doctor": alternative,
                    "emergency_doctors_available": emergency_doctors,
                    "can_book_alternative": bool(alternative),
                }
            )

        if fa._is_doctor_on_leave(doctor_id, parsed_time):
            if risk_level_text == "high":
                emergency_doctors = fa._available_emergency_doctors(parsed_time, exclude_doctor_id=doctor_id)
                alternative = emergency_doctors[0] if emergency_doctors else None
                if alternative:
                    return jsonify(
                        {
                            "booking_status": "doctor_unavailable",
                            "reason": "doctor_on_leave",
                            "risk_level": "High",
                            "message": "Selected doctor is on leave. Because your condition is high risk, we recommend consulting an available emergency doctor.",
                            "selected_doctor": selected_doctor_profile,
                            "alternative_doctor": alternative,
                            "emergency_doctors_available": emergency_doctors,
                            "can_book_alternative": True,
                        }
                    )
                return jsonify(
                    {
                        "booking_status": "doctor_unavailable",
                        "reason": "doctor_on_leave",
                        "risk_level": "High",
                        "message": "Selected doctor is on leave. Because your condition is high risk, no emergency doctor is currently available on this date.",
                        "selected_doctor": selected_doctor_profile,
                        "alternative_doctor": None,
                        "emergency_doctors_available": [],
                        "can_book_alternative": False,
                    }
                )

            return jsonify(
                {
                    "booking_status": "doctor_unavailable",
                    "reason": "doctor_on_leave",
                    "risk_level": risk_level_text.title() or "Low",
                    "message": "Selected doctor is on leave. Please choose another date or another doctor.",
                    "selected_doctor": selected_doctor_profile,
                    "alternative_doctor": None,
                    "emergency_doctors_available": [],
                    "can_book_alternative": False,
                }
            )

        priority = fa.determine_priority(risk_assessment.risk_level)
        appointment_id = fa.uuid4().hex
        result = {
            "booking_status": "confirmed",
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "patient_name": patient_name,
            "contact_info": normalized_contact,
            "doctor_id": doctor_id,
            "appointment_type": appointment_type,
            "appointment_time": parsed_time.isoformat(),
            "predicted_disease": risk_assessment.predicted_class,
            "risk_level": risk_assessment.risk_level.title(),
            "appointment_priority": priority.priority,
            "recommended_slot": priority.recommended_slot,
            "priority_badge_text": priority.badge_text,
            "priority_badge_color": priority.badge_color,
            "redirect_url": url_for("patient_booking_confirmation"),
        }
        fa.patient_db.add_appointment(
            {
                "appointment_id": appointment_id,
                "booked_at": fa.datetime.now(fa.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "patient_id": patient_id,
                "patient_name": patient_name,
                "contact_info": normalized_contact,
                "doctor_id": doctor_id,
                "appointment_type": appointment_type,
                "appointment_time": parsed_time.isoformat(),
                "patient_features": fa.to_jsonable(patient_features),
                "risk_assessment": risk_assessment.model_dump(),
                "appointment_priority": priority.priority,
                "recommended_slot": priority.recommended_slot,
                "priority_badge_text": priority.badge_text,
                "priority_badge_color": priority.badge_color,
            }
        )
        return jsonify(result)

    @app.post("/patient/predict-risk")
    @fa.patient_required(api=True)
    def patient_predict_risk() -> Any:
        payload = request.get_json(silent=True) or {}
        patient_features = payload.get("patient_features")
        if not patient_features:
            return jsonify({"detail": "patient_features is required"}), 400

        try:
            result = fa.risk_engine.predict(patient_features)
            priority = fa.determine_priority(result.risk_level)
            response_payload = result.model_dump()
            response_payload.update(
                {
                    "appointment_priority": priority.priority,
                    "recommended_slot": priority.recommended_slot,
                    "priority_badge_text": priority.badge_text,
                    "priority_badge_color": priority.badge_color,
                }
            )
            return jsonify(response_payload)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - guard
            fa.logger.exception("Patient portal risk prediction failed for patient_id=%s", session.get("patient_id"))
            return jsonify({"detail": f"Prediction failed: {exc}"}), 500

    @app.get("/patient/emergency-doctors")
    @fa.patient_required(api=True)
    def patient_emergency_doctors() -> Any:
        date_text = str(request.args.get("date", "")).strip()
        time_text = str(request.args.get("time", "")).strip()
        if not date_text:
            return jsonify({"detail": "date is required (YYYY-MM-DD)"}), 400
        if not time_text:
            return jsonify({"detail": "time is required (HH:MM)"}), 400
        try:
            parsed = fa.datetime.fromisoformat(f"{date_text}T{time_text}")
        except ValueError:
            return jsonify({"detail": "Invalid date/time format"}), 400

        is_night_window = fa._is_night_emergency_window(parsed)
        doctors = fa._available_emergency_doctors(parsed) if is_night_window else []
        return jsonify(
            {
                "date": parsed.date().isoformat(),
                "time": f"{parsed.hour:02d}:{parsed.minute:02d}",
                "window_active": is_night_window,
                "emergency_doctors": doctors,
            }
        )
