"""
pipeline/audio_compressor.py

Converts audio to the optimal format for WhisperX: 16kHz, mono, 16-bit WAV.
Can prepend silence to bake in a fixed audio-to-video offset, so WhisperX
timestamps align directly with video time without post-processing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

# Files already at or below this size don't need conversion
COMPRESSION_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB

# Default offset: audio recording starts this many seconds into the video
DEFAULT_AUDIO_OFFSET_SECONDS = 9.13


def prepare_audio_with_offset(
    audio_file_path: Path,
    offset_seconds: float = DEFAULT_AUDIO_OFFSET_SECONDS,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Convert audio to 16kHz/mono/16-bit WAV AND prepend silence equal to
    offset_seconds. This makes WhisperX timestamps already aligned to
    video time — no post-processing shift needed.

    Always converts regardless of file size.
    Caches as {stem}.offset.16k.wav. Returns path to the prepared file.
    """
    def _log(message: str) -> None:
        if on_progress:
            on_progress(message)

    converted_path = audio_file_path.parent / f"{audio_file_path.stem}.offset.16k.wav"

    if converted_path.exists():
        converted_size_megabytes = converted_path.stat().st_size / (1024 * 1024)
        _log(f"Using cached offset audio ({converted_size_megabytes:.0f} MB)")
        return converted_path

    file_size_megabytes = audio_file_path.stat().st_size / (1024 * 1024)
    delay_milliseconds = int(offset_seconds * 1000)

    _log(
        f"Converting {audio_file_path.name} ({file_size_megabytes:.0f} MB) "
        f"to 16kHz/16-bit/mono WAV with {offset_seconds}s silence prepended..."
    )

    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", str(audio_file_path),
        "-vn",
        "-af", f"adelay={delay_milliseconds}|{delay_milliseconds}",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(converted_path),
    ]

    result = subprocess.run(
        ffmpeg_command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[-500:]}")

    converted_size_megabytes = converted_path.stat().st_size / (1024 * 1024)
    _log(
        f"Converted: {file_size_megabytes:.0f} MB -> {converted_size_megabytes:.0f} MB "
        f"(16kHz/16-bit/mono + {offset_seconds}s offset)"
    )

    return converted_path


def compress_audio_if_needed(
    audio_file_path: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Convert audio to 16kHz mono 16-bit WAV (what Whisper wants internally).
    Only converts files above 50 MB. No offset prepended.
    Returns the path to use for upload.
    """
    def _log(message: str) -> None:
        if on_progress:
            on_progress(message)

    file_size = audio_file_path.stat().st_size
    file_size_megabytes = file_size / (1024 * 1024)

    if file_size < COMPRESSION_THRESHOLD_BYTES:
        _log(f"Audio is small ({file_size_megabytes:.0f} MB), skipping conversion")
        return audio_file_path

    converted_path = audio_file_path.with_suffix(".16k.wav")

    if converted_path.exists():
        converted_size_megabytes = converted_path.stat().st_size / (1024 * 1024)
        _log(f"Using existing converted file ({converted_size_megabytes:.0f} MB)")
        return converted_path

    _log(f"Audio is {file_size_megabytes:.0f} MB -- converting to 16kHz/16-bit/mono WAV...")

    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", str(audio_file_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(converted_path),
    ]

    result = subprocess.run(
        ffmpeg_command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[-500:]}")

    converted_size_megabytes = converted_path.stat().st_size / (1024 * 1024)
    _log(f"Converted: {file_size_megabytes:.0f} MB -> {converted_size_megabytes:.0f} MB (16kHz/16-bit/mono)")

    return converted_path
