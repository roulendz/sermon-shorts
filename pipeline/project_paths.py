"""
pipeline/project_paths.py

Single source of truth for the on-disk layout of an event folder.

The "event root" is the directory that contains the selected video file
(e.g. ``U:\\2026-06-07 Kā atnāk Dieva valstība``). Every artifact the
pipeline produces lives under a well-known subfolder so the root stays clean:

    <event root>/
        Audio RAW/             raw per-language audio tracks (input)
        Audio TRANSCRIPT/      all transcription artifacts
            transcriptions/    service .srt files (downstream-compatible)
            whisperx/v{N}/      raw versioned WhisperX JSON responses
        Clips RAW/             rendered clips (+ v{N}-{date} subfolders)
        .data/                 machine state (Manus responses + task registry)

Import these helpers everywhere instead of hard-coding folder names so the
layout can change in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

# ── Folder names ─────────────────────────────────────────────────────────────
AUDIO_RAW_FOLDER_NAME = "Audio RAW"
AUDIO_TRANSCRIPT_FOLDER_NAME = "Audio TRANSCRIPT"
TRANSCRIPTIONS_FOLDER_NAME = "transcriptions"
WHISPERX_STORAGE_FOLDER_NAME = "whisperx"
CLIPS_FOLDER_NAME = "Clips RAW"
DATA_FOLDER_NAME = ".data"


def event_root_directory(video_file_path: Path) -> Path:
    """Directory that contains the selected video file."""
    return video_file_path.parent


def _resolve(video_file_path: Path, *parts: str, create: bool) -> Path:
    directory = event_root_directory(video_file_path).joinpath(*parts)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def audio_transcript_directory(video_file_path: Path, create: bool = False) -> Path:
    """``<event root>/Audio TRANSCRIPT`` — parent of all transcript artifacts."""
    return _resolve(video_file_path, AUDIO_TRANSCRIPT_FOLDER_NAME, create=create)


def transcriptions_directory(video_file_path: Path, create: bool = False) -> Path:
    """``<event root>/Audio TRANSCRIPT/transcriptions`` — service .srt files."""
    return _resolve(
        video_file_path,
        AUDIO_TRANSCRIPT_FOLDER_NAME,
        TRANSCRIPTIONS_FOLDER_NAME,
        create=create,
    )


def whisperx_storage_directory(video_file_path: Path, create: bool = False) -> Path:
    """``<event root>/Audio TRANSCRIPT/whisperx`` — root of versioned responses."""
    return _resolve(
        video_file_path,
        AUDIO_TRANSCRIPT_FOLDER_NAME,
        WHISPERX_STORAGE_FOLDER_NAME,
        create=create,
    )


def clips_directory(video_file_path: Path, create: bool = False) -> Path:
    """``<event root>/Clips RAW`` — rendered clips and v{N}-{date} subfolders."""
    return _resolve(video_file_path, CLIPS_FOLDER_NAME, create=create)


def data_directory(video_file_path: Path, create: bool = False) -> Path:
    """``<event root>/.data`` — Manus responses and task registry."""
    return _resolve(video_file_path, DATA_FOLDER_NAME, create=create)
