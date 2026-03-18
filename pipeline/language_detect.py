"""
pipeline/language_detect.py

Detects language from audio filename suffix.
_A03 = Latvian, _A04 = Russian, _A05 = English.

Provides locale mapping for services that need BCP-47 codes (e.g. Transkriptor).
"""

from __future__ import annotations

from pathlib import Path

TRACK_LANGUAGE_MAP = {
    "_A03": "lv",
    "_A04": "ru",
    "_A05": "en",
}

LANGUAGE_NAMES = {
    "lv": "Latvian",
    "ru": "Russian",
    "en": "English",
}

# WhisperX uses short codes, Transkriptor uses BCP-47 locale codes
TRANSKRIPTOR_LOCALE_MAP = {
    "lv": "lv-LV",
    "ru": "ru-RU",
    "en": "en-US",
}

DEFAULT_LANGUAGE = "lv"


def detect_language_from_filename(audio_file_path: Path) -> str:
    """Return language code based on audio filename suffix."""
    stem = audio_file_path.stem  # e.g. "2026-01-04 Pacelot mīlestības karogu_A03"
    for suffix, lang_code in TRACK_LANGUAGE_MAP.items():
        if suffix in stem:
            return lang_code
    return DEFAULT_LANGUAGE


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def to_transkriptor_locale(language_code: str) -> str:
    """Convert short language code to Transkriptor BCP-47 locale."""
    return TRANSKRIPTOR_LOCALE_MAP.get(language_code, f"{language_code}-{language_code.upper()}")
