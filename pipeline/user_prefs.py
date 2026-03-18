"""
pipeline/user_prefs.py

Persists user preferences (e.g. last browsed folder) across runs.
Stored as a JSON file in the project directory.
"""

from __future__ import annotations

import json
from pathlib import Path

PREFS_FILE = Path(__file__).parent.parent / ".user_prefs.json"


def _load() -> dict:
    if PREFS_FILE.exists():
        try:
            return json.loads(PREFS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(data: dict) -> None:
    PREFS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_last_browse_directory() -> Path | None:
    path_str = _load().get("last_browse_directory")
    if path_str:
        p = Path(path_str)
        if p.exists():
            return p
    return None


def set_last_browse_directory(directory: Path) -> None:
    data = _load()
    data["last_browse_directory"] = str(directory)
    _save(data)
