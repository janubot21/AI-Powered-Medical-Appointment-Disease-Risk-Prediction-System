from __future__ import annotations

from typing import Any, Dict

from flask import jsonify, redirect, render_template, request, session, url_for


def register_staff_routes(app: Any) -> None:
    import flask_app as fa

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
            if not fa.PASSWORD_POLICY_PATTERN.fullmatch(password):
                errors.append(fa.PASSWORD_POLICY_MESSAGE)
            if password != confirm_password:
                errors.append("Password and Confirm Password must match.")

            if not errors:
                try:
                    fa.doctor_auth_manager.signup(doctor_id, password)
                    return redirect(url_for("doctor_login"))
                except ValueError as exc:
                    errors.append(str(exc))
                except Exception as exc:  # pragma: no cover - guard
                    fa.logger.exception("Doctor portal signup failed for doctor_id=%s", doctor_id)
                    errors.append(f"Signup failed: {exc}")

        return render_template("flask_doctor_signup.html", errors=errors, form_data=form_data)

    @app.route("/doctor/login", methods=["GET", "POST"])
    def doctor_login() -> Any:
        errors: list[str] = []
        form_data = {"doctor_id": ""}

        if request.method == "GET":
            fa._clear_doctor_session()
            if session.get("role") == "doctor":
                session.pop("role", None)

        if request.method == "POST":
            doctor_id = request.form.get("doctor_id", "").strip()
            password = request.form.get("password", "").strip()
            form_data["doctor_id"] = doctor_id
            if not doctor_id:
                errors.append("Enter Doctor ID.")
            if not password:
                errors.append("Password is required.")

            if not errors:
                try:
                    resolved_doctor_id = fa.doctor_auth_manager.login(doctor_id, "", "", password)
                    fa._clear_patient_session()
                    fa._clear_nurse_session()
                    session["doctor_id"] = resolved_doctor_id
                    session["doctor_name"] = resolved_doctor_id
                    session["doctor_authenticated"] = True
                    session["role"] = "doctor"
                    return render_template(
                        "flask_doctor_login.html",
                        errors=[],
                        form_data={"doctor_id": ""},
                        login_success=True,
                        redirect_url=url_for("doctor_dashboard"),
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                except Exception as exc:  # pragma: no cover - guard
                    fa.logger.exception("Doctor portal login failed for doctor_id=%s", doctor_id)
                    errors.append(f"Login failed: {exc}")

        return render_template("flask_doctor_login.html", errors=errors, form_data=form_data)

    @app.route("/nurse/signup", methods=["GET", "POST"])
    def nurse_signup() -> Any:
        errors: list[str] = []
        form_data = {"nurse_id": ""}

        if request.method == "POST":
            nurse_id = request.form.get("nurse_id", "").strip()
            password = request.form.get("password", "").strip()
            confirm_password = request.form.get("confirm_password", "").strip()
            form_data["nurse_id"] = nurse_id

            if not nurse_id:
                errors.append("Nurse ID is required.")
            if not fa.PASSWORD_POLICY_PATTERN.fullmatch(password):
                errors.append(fa.PASSWORD_POLICY_MESSAGE)
            if password != confirm_password:
                errors.append("Password and Confirm Password must match.")

            if not errors:
                try:
                    fa.nurse_auth_manager.signup(nurse_id, password)
                    return redirect(url_for("nurse_login"))
                except ValueError as exc:
                    errors.append(str(exc))
                except Exception as exc:  # pragma: no cover - guard
                    fa.logger.exception("Nurse portal signup failed for nurse_id=%s", nurse_id)
                    errors.append(f"Signup failed: {exc}")

        return render_template("flask_nurse_signup.html", errors=errors, form_data=form_data)

    @app.route("/nurse/login", methods=["GET", "POST"])
    def nurse_login() -> Any:
        errors: list[str] = []
        form_data = {"nurse_id": ""}

        if request.method == "GET":
            fa._clear_nurse_session()
            if session.get("role") == "nurse":
                session.pop("role", None)

        if request.method == "POST":
            nurse_id = request.form.get("nurse_id", "").strip()
            password = request.form.get("password", "").strip()
            form_data["nurse_id"] = nurse_id

            if not nurse_id:
                errors.append("Enter Nurse ID.")
            if not password:
                errors.append("Password is required.")

            if not errors:
                try:
                    resolved_nurse_id = fa.nurse_auth_manager.login(nurse_id, password)
                    fa._clear_patient_session()
                    fa._clear_doctor_session()
                    session["nurse_id"] = resolved_nurse_id
                    session["nurse_name"] = resolved_nurse_id
                    session["nurse_authenticated"] = True
                    session["role"] = "nurse"
                    return render_template(
                        "flask_nurse_login.html",
                        errors=[],
                        form_data={"nurse_id": ""},
                        login_success=True,
                        redirect_url=url_for("nurse_dashboard"),
                    )
                except ValueError as exc:
                    errors.append(str(exc))
                except Exception as exc:  # pragma: no cover - guard
                    fa.logger.exception("Nurse portal login failed for nurse_id=%s", nurse_id)
                    errors.append(f"Login failed: {exc}")

        return render_template("flask_nurse_login.html", errors=errors, form_data=form_data)

    @app.route("/nurse/dashboard")
    @fa.nurse_required()
    def nurse_dashboard() -> Any:
        nurse_id = session.get("nurse_id")
        nurse_name = str(session.get("nurse_name", "")).strip() or str(nurse_id)
        return render_template("flask_nurse_dashboard.html", nurse_id=nurse_id, nurse_name=nurse_name)

    @app.route("/nurse/appointment-queue")
    @fa.nurse_required()
    def nurse_appointment_queue() -> Any:
        nurse_id = session.get("nurse_id")
        nurse_name = str(session.get("nurse_name", "")).strip() or str(nurse_id)
        return render_template("flask_nurse_queue.html", nurse_id=nurse_id, nurse_name=nurse_name)

    @app.get("/nurse/patient-records")
    @fa.nurse_required(api=True)
    def nurse_patient_records() -> Any:
        return jsonify({"patients": fa._nurse_patient_records()})

    @app.post("/nurse/patient-records")
    @fa.nurse_required(api=True)
    def nurse_update_patient_record() -> Any:
        payload = request.get_json(silent=True) or {}
        nurse_id = str(session.get("nurse_id", "")).strip()
        patient_id = fa._normalize_patient_id(payload.get("patient_id"))
        if not patient_id:
            return jsonify({"detail": "patient_id is required"}), 400
        try:
            record = fa._nurse_record_payload(payload, nurse_id)
            patient_features = fa._nurse_patient_features(record)
            risk_report = fa._build_risk_report(patient_id, patient_features)
            fa.upsert_new_patient_csv_from_features(patient_id, patient_features)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - guard
            fa.logger.exception("Could not save nurse record for patient_id=%s", patient_id)
            return jsonify({"detail": f"Could not save nurse record: {exc}"}), 500

        saved_profile = fa.patient_db.get_profile(patient_id)
        return jsonify({"status": "ok", "patient": saved_profile, "risk_report": risk_report})

    @app.post("/nurse/predict-risk")
    @fa.nurse_required(api=True)
    def nurse_predict_risk() -> Any:
        payload = request.get_json(silent=True) or {}
        nurse_id = str(session.get("nurse_id", "")).strip()
        patient_id = fa._normalize_patient_id(payload.get("patient_id"))
        if not patient_id:
            return jsonify({"detail": "patient_id is required"}), 400

        try:
            record = fa._nurse_record_payload(payload, nurse_id)
            patient_features = fa._nurse_patient_features(record)
            risk_report = fa._build_risk_report(patient_id, patient_features)
            return jsonify({"patient_id": patient_id, "risk_report": risk_report})
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        except Exception as exc:  # pragma: no cover - guard
            fa.logger.exception("Nurse risk prediction failed for patient_id=%s", patient_id)
            return jsonify({"detail": f"Prediction failed: {exc}"}), 500

    @app.route("/nurse/logout")
    def nurse_logout() -> Any:
        fa._clear_nurse_session()
        if session.get("role") == "nurse":
            session.pop("role", None)
        return redirect(url_for("role_login"))

    @app.route("/doctor/dashboard")
    @fa.doctor_required()
    def doctor_dashboard() -> Any:
        doctor_id = session.get("doctor_id")
        doctor_name = str(session.get("doctor_name", "")).strip() or str(doctor_id)
        return render_template("flask_doctor_dashboard.html", doctor_id=doctor_id, doctor_name=doctor_name)

    @app.route("/doctor/past-dashboard")
    @fa.doctor_required()
    def doctor_past_dashboard() -> Any:
        doctor_id = session.get("doctor_id")
        doctor_name = str(session.get("doctor_name", "")).strip() or str(doctor_id)
        return render_template("flask_doctor_past_dashboard.html", doctor_id=doctor_id, doctor_name=doctor_name)

    @app.route("/doctor/patient-database-page")
    @fa.doctor_required()
    def doctor_patient_database_page() -> Any:
        doctor_id = session.get("doctor_id")
        doctor_name = str(session.get("doctor_name", "")).strip() or str(doctor_id)
        return render_template(
            "flask_doctor_patient_database.html",
            doctor_id=doctor_id,
            doctor_name=doctor_name,
        )

    @app.route("/doctor/logout")
    def doctor_logout() -> Any:
        session.pop("doctor_id", None)
        session.pop("doctor_name", None)
        session.pop("doctor_authenticated", None)
        if session.get("role") == "doctor":
            session.pop("role", None)
        return redirect(url_for("role_login"))

    @app.get("/doctor/appointments")
    @fa.doctor_required(api=True)
    def doctor_appointments() -> Any:
        current_doctor_id = fa._normalize_doctor_id(session.get("doctor_id"))
        appointments = [
            appointment
            for appointment in fa.patient_db.list_appointments()
            if fa._normalize_doctor_id(appointment.get("doctor_id")) == current_doctor_id
        ]
        appointments = sorted(appointments, key=fa._doctor_appointment_sort_key)
        return jsonify({"appointments": appointments})

    @app.get("/doctor/patient-database")
    @fa.doctor_required(api=True)
    def doctor_patient_database() -> Any:
        current_doctor_id = fa._normalize_doctor_id(session.get("doctor_id"))
        all_profiles = fa.patient_db.list_profiles()
        doctor_appointments = [
            appointment
            for appointment in fa.patient_db.list_appointments()
            if fa._normalize_doctor_id(appointment.get("doctor_id")) == current_doctor_id
        ]
        appointment_lookup: Dict[str, Dict[str, Any]] = {}
        for appointment in doctor_appointments:
            pid = fa._normalize_patient_id(appointment.get("patient_id"))
            if pid and pid not in appointment_lookup:
                appointment_lookup[pid] = appointment

        doctor_patient_ids = {
            fa._normalize_patient_id(ap.get("patient_id"))
            for ap in doctor_appointments
            if fa._normalize_patient_id(ap.get("patient_id"))
        }

        patients = []
        source_rows = [row for row in all_profiles if fa._normalize_patient_id(row.get("patient_id")) in doctor_patient_ids]
        for row in source_rows:
            patient = dict(row)
            appointment = appointment_lookup.get(fa._normalize_patient_id(patient.get("patient_id")))
            profile_features = fa._patient_features_from_profile(patient)
            appointment_features = dict((appointment or {}).get("patient_features") or {})
            merged_features = {
                **appointment_features,
                **{
                    key: value
                    for key, value in profile_features.items()
                    if value is not None and value != "" and value != []
                },
            }
            patient["patient_features"] = merged_features
            try:
                patient["risk_report"] = fa._build_risk_report(
                    str(patient.get("patient_id", "")).strip(),
                    merged_features,
                )
            except Exception:
                assessment = (appointment or {}).get("risk_assessment") or {}
                patient["risk_report"] = (
                    {
                        **assessment,
                        "appointment_priority": (appointment or {}).get("appointment_priority"),
                        "recommended_slot": (appointment or {}).get("recommended_slot"),
                        "priority_badge_text": (appointment or {}).get("priority_badge_text"),
                        "priority_badge_color": (appointment or {}).get("priority_badge_color"),
                    }
                    if assessment
                    else {}
                )
            patients.append(patient)
        patients.sort(
            key=lambda item: (
                -fa._datetime_sort_value(fa._parse_appointment_time(item.get("health_data_submitted_at")), default=0.0),
                -fa._datetime_sort_value(fa._parse_appointment_time(item.get("updated_at")), default=0.0),
                (0, int(str(item.get("patient_id")))) if str(item.get("patient_id")).isdigit() else (1, str(item.get("patient_id"))),
            )
        )
        return jsonify({"patients": patients})

    @app.get("/doctor/leaves")
    @fa.doctor_required(api=True)
    def doctor_leaves() -> Any:
        doctor_id = str(request.args.get("doctor_id", "")).strip()
        rows = fa._load_doctor_leave_df().to_dict(orient="records")
        if doctor_id:
            doctor_norm = fa._normalize_doctor_id(doctor_id)
            rows = [r for r in rows if fa._normalize_doctor_id(r.get("doctor_id", "")) == doctor_norm]
        return jsonify({"leaves": rows})

    @app.post("/doctor/leaves")
    @fa.doctor_required(api=True)
    def upsert_doctor_leave() -> Any:
        payload = request.get_json(silent=True) or {}
        leave_date = str(payload.get("leave_date", "")).strip()
        doctor_id = str(payload.get("doctor_id", "")).strip() or str(session.get("doctor_id", "")).strip()
        if not doctor_id:
            return jsonify({"detail": "doctor_id is required"}), 400
        if not leave_date:
            return jsonify({"detail": "leave_date is required"}), 400
        try:
            fa._add_doctor_leave(doctor_id, leave_date)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        return jsonify({"status": "ok", "doctor_id": doctor_id, "leave_date": leave_date[:10]})

    @app.get("/doctor/availability-status")
    @fa.doctor_required(api=True)
    def doctor_availability_status() -> Any:
        doctor_id = str(request.args.get("doctor_id", "")).strip() or str(session.get("doctor_id", "")).strip()
        if not doctor_id:
            return jsonify({"detail": "doctor_id is required"}), 400
        date_text = str(request.args.get("date", "")).strip()
        date_key = date_text[:10] if date_text else fa.datetime.now().date().isoformat()
        try:
            parsed = fa.datetime.fromisoformat(date_key)
        except ValueError:
            return jsonify({"detail": "date must be YYYY-MM-DD"}), 400
        day = parsed.date().isoformat()
        status = "leave" if fa._is_doctor_on_leave(doctor_id, parsed) else "available"
        return jsonify({"doctor_id": doctor_id, "date": day, "status": status})

    @app.post("/doctor/availability-status")
    @fa.doctor_required(api=True)
    def set_doctor_availability_status() -> Any:
        payload = request.get_json(silent=True) or {}
        doctor_id = str(payload.get("doctor_id", "")).strip() or str(session.get("doctor_id", "")).strip()
        if not doctor_id:
            return jsonify({"detail": "doctor_id is required"}), 400
        status = str(payload.get("status", "")).strip().lower()
        if status not in {"available", "leave"}:
            return jsonify({"detail": "status must be 'available' or 'leave'"}), 400
        date_text = str(payload.get("date", "")).strip()
        date_key = date_text[:10] if date_text else fa.datetime.now().date().isoformat()
        try:
            parsed = fa.datetime.fromisoformat(date_key)
        except ValueError:
            return jsonify({"detail": "date must be YYYY-MM-DD"}), 400
        day = parsed.date().isoformat()
        try:
            if status == "leave":
                fa._add_doctor_leave(doctor_id, day)
            else:
                fa._remove_doctor_leave(doctor_id, day)
        except ValueError as exc:
            return jsonify({"detail": str(exc)}), 400
        return jsonify({"doctor_id": doctor_id, "date": day, "status": status})

    @app.post("/doctor/predict-risk")
    @fa.doctor_required(api=True)
    def doctor_predict_risk() -> Any:
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
            fa.logger.exception("Doctor portal risk prediction failed for doctor_id=%s", session.get("doctor_id"))
            return jsonify({"detail": f"Prediction failed: {exc}"}), 500
