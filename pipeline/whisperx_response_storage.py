"""
pipeline/whisperx_response_storage.py

Versioned storage for raw WhisperX API responses.
Stores full JSON (including word-level data) in:
    {video_directory}/whisperx/v{N}/{audio_stem}.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

WHISPERX_STORAGE_FOLDER_NAME = "whisperx"
VERSION_DIRECTORY_PATTERN = re.compile(r"^v(\d+)$")


def find_latest_version_number(video_file_path: Path) -> Optional[int]:
    """
    Scan {video_dir}/whisperx/ for versioned folders (v1, v2, ...).
    Returns the highest version number found, or None if none exist.
    """
    whisperx_directory = video_file_path.parent / WHISPERX_STORAGE_FOLDER_NAME

    if not whisperx_directory.is_dir():
        return None

    highest_version = None
    for item in whisperx_directory.iterdir():
        if item.is_dir():
            match = VERSION_DIRECTORY_PATTERN.match(item.name)
            if match:
                version_number = int(match.group(1))
                if highest_version is None or version_number > highest_version:
                    highest_version = version_number

    return highest_version


def build_version_directory_path(
    video_file_path: Path,
    version_number: int,
) -> Path:
    """Return the path to a specific version directory."""
    return (
        video_file_path.parent
        / WHISPERX_STORAGE_FOLDER_NAME
        / f"v{version_number}"
    )


def build_next_version_directory_path(video_file_path: Path) -> Path:
    """
    Determine the path for the next version directory.
    If no versions exist, returns v1/. Otherwise returns v(N+1)/.
    """
    latest_version = find_latest_version_number(video_file_path)
    next_version = 1 if latest_version is None else latest_version + 1
    return build_version_directory_path(video_file_path, next_version)


def save_whisperx_response(
    video_file_path: Path,
    audio_file_stem: str,
    response_data: dict[str, Any],
    version_directory: Path,
) -> Path:
    """
    Save a WhisperX JSON response to the given version directory.
    Creates the directory if needed.
    Returns the path to the saved JSON file.
    """
    version_directory.mkdir(parents=True, exist_ok=True)
    output_path = version_directory / f"{audio_file_stem}.json"
    output_path.write_text(
        json.dumps(response_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"WhisperX response saved: {output_path}")
    return output_path


def load_whisperx_response(
    video_file_path: Path,
    audio_file_stem: str,
    version_number: int,
) -> dict[str, Any]:
    """
    Load a stored WhisperX JSON response for a specific version.
    Raises FileNotFoundError if the file doesn't exist.
    """
    version_directory = build_version_directory_path(video_file_path, version_number)
    response_path = version_directory / f"{audio_file_stem}.json"
    raw_text = response_path.read_text(encoding="utf-8")
    return json.loads(raw_text)


def list_stored_audio_stems_for_version(
    video_file_path: Path,
    version_number: int,
) -> list[str]:
    """
    List all audio stems that have stored responses in a given version.
    E.g. returns ["sermon_A03", "sermon_A04"] for version 1.
    """
    version_directory = build_version_directory_path(video_file_path, version_number)
    if not version_directory.is_dir():
        return []
    return [
        json_file.stem
        for json_file in version_directory.glob("*.json")
    ]
