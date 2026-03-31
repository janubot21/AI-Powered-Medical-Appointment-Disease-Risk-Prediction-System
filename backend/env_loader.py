from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _candidate_env_files() -> list[Path]:
    primary = PROJECT_ROOT / ".env"
    return [primary] if primary.exists() else []


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env() -> None:
    for env_file in _candidate_env_files():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            env_key = key.strip()
            if not env_key or env_key in os.environ:
                continue
            os.environ[env_key] = _parse_env_value(value)


load_project_env()
