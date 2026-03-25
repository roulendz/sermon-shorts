"""
pipeline/audio_track_discovery.py

Auto-discovers audio track files from the "Audio RAW" subfolder
adjacent to the video file. Maps track suffixes (_A03, _A04, _A05)
to language codes using TRACK_LANGUAGE_MAP.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline.language_detect import TRACK_LANGUAGE_MAP, LANGUAGE_NAMES

logger = logging.getLogger(__name__)

AUDIO_RAW_FOLDER_NAME = "Audio RAW"
SUPPORTED_AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".flac"}


def discover_audio_tracks(video_file_path: Path) -> dict[str, Path]:
    """
    Look for audio files in the "Audio RAW" subfolder next to the video.
    Returns a dict mapping language code (e.g. "lv", "ru") to the audio file path.
    Returns an empty dict if no Audio RAW folder or no matching files.
    """
    video_directory = video_file_path.parent
    audio_raw_directory = video_directory / AUDIO_RAW_FOLDER_NAME

    if not audio_raw_directory.is_dir():
        logger.info(f"Audio RAW folder not found: {audio_raw_directory}")
        return {}

    discovered_tracks: dict[str, Path] = {}

    for audio_file in audio_raw_directory.iterdir():
        if not audio_file.is_file():
            continue
        if audio_file.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue

        file_stem = audio_file.stem
        for track_suffix, language_code in TRACK_LANGUAGE_MAP.items():
            if file_stem.endswith(track_suffix):
                discovered_tracks[language_code] = audio_file
                logger.info(
                    f"Discovered {LANGUAGE_NAMES.get(language_code, language_code)} "
                    f"track: {audio_file.name}"
                )
                break

    return discovered_tracks


def format_discovered_tracks_summary(discovered_tracks: dict[str, Path]) -> str:
    """
    Build a human-readable summary of discovered tracks.
    e.g. "Found: Latvian (_A03), Russian (_A04)"
    """
    if not discovered_tracks:
        return "No audio tracks found in Audio RAW folder"

    language_to_suffix = {value: key for key, value in TRACK_LANGUAGE_MAP.items()}

    parts = []
    for language_code in sorted(discovered_tracks.keys()):
        suffix = language_to_suffix.get(language_code, "")
        language_display_name = LANGUAGE_NAMES.get(language_code, language_code)
        parts.append(f"{language_display_name} ({suffix})")

    return f"Found: {', '.join(parts)}"
