from __future__ import annotations

import os
import secrets

from env_loader import load_project_env


load_project_env()

_TRUTHY_VALUES = {"1", "true", "yes", "on"}


def env_text(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return str(value).strip()


def env_flag(name: str, default: bool = False) -> bool:
    fallback = "1" if default else "0"
    return env_text(name, fallback).lower() in _TRUTHY_VALUES


def get_flask_secret_key() -> str:
    for key_name in ("FLASK_SECRET_KEY", "SECRET_KEY"):
        value = env_text(key_name)
        if value:
            return value
    return secrets.token_hex(32)


COOKIE_SECURE = env_flag("COOKIE_SECURE", default=False)
OPENAI_API_KEY = env_text("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = env_text("OPENAI_CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
