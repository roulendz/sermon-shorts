"""
pipeline/audio_compressor.py

Converts audio to the optimal format for WhisperX: 16kHz, mono, 16-bit WAV.
Whisper internally resamples everything to this anyway, so pre-converting
saves upload time without any accuracy loss.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

# Files already at or below this size don't need conversion
COMPRESSION_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB


def compress_audio_if_needed(
    audio_file_path: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Convert audio to 16kHz mono 16-bit WAV (what Whisper wants internally).
    Returns the path to use for upload.
    """
    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    file_size = audio_file_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    if file_size < COMPRESSION_THRESHOLD_BYTES:
        _log(f"Audio is small ({file_size_mb:.0f} MB), skipping conversion")
        return audio_file_path

    converted_path = audio_file_path.with_suffix(".16k.wav")

    if converted_path.exists():
        converted_size_mb = converted_path.stat().st_size / (1024 * 1024)
        _log(f"Using existing converted file ({converted_size_mb:.0f} MB)")
        return converted_path

    _log(f"Audio is {file_size_mb:.0f} MB -- converting to 16kHz/16-bit/mono WAV...")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_file_path),
        "-vn",                     # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM (what Whisper uses internally)
        "-ar", "16000",            # 16kHz sample rate
        "-ac", "1",                # mono
        str(converted_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[-500:]}")

    converted_size_mb = converted_path.stat().st_size / (1024 * 1024)
    _log(f"Converted: {file_size_mb:.0f} MB -> {converted_size_mb:.0f} MB (16kHz/16-bit/mono)")

    return converted_path
