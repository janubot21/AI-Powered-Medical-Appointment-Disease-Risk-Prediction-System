from __future__ import annotations

from typing import Any

from flask import redirect, render_template, url_for


def register_common_routes(app: Any) -> None:
    @app.route("/")
    def home() -> Any:
        return redirect(url_for("role_login"))

    @app.route("/login")
    def role_login() -> Any:
        return render_template("flask_role_login.html")

    @app.route("/about")
    def about_page() -> Any:
        return render_template("flask_about.html")
