"""
pipeline/audio_compressor.py

Audio format conversion for transcription services.
- WhisperX: 16kHz mono 16-bit WAV (with optional offset prepend)
- Transkriptor: 16kHz mono MP3 128kbps (small upload size)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

COMPRESSION_THRESHOLD_BYTES = 50 * 1024 * 1024  # 50 MB

DEFAULT_AUDIO_OFFSET_SECONDS = 9.13

TRANSKRIPTOR_BITRATE = "128k"
TRANSKRIPTOR_SAMPLE_RATE = "16000"


def _run_ffmpeg_conversion(
    ffmpeg_arguments: list[str],
    output_file_path: Path,
) -> None:
    logger.debug("FFmpeg command: %s", " ".join(ffmpeg_arguments))
    result = subprocess.run(
        ffmpeg_arguments,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        logger.error("FFmpeg conversion failed (exit %d): %s", result.returncode, result.stderr[-500:])
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr[-500:]}")
    logger.info("FFmpeg conversion complete: %s", output_file_path)


def _format_file_size_megabytes(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)


def prepare_audio_with_offset(
    audio_file_path: Path,
    offset_seconds: float = DEFAULT_AUDIO_OFFSET_SECONDS,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Convert audio to 16kHz/mono/16-bit WAV with silence prepended for offset.
    Caches as {stem}.offset.16k.wav.
    """
    def _log(message: str) -> None:
        logger.info(message)
        if on_progress:
            on_progress(message)

    converted_path = audio_file_path.parent / f"{audio_file_path.stem}.offset.16k.wav"

    if converted_path.exists():
        _log(f"Using cached offset audio ({_format_file_size_megabytes(converted_path):.0f} MB)")
        return converted_path

    source_size_megabytes = _format_file_size_megabytes(audio_file_path)
    delay_milliseconds = int(offset_seconds * 1000)

    _log(
        f"Converting {audio_file_path.name} ({source_size_megabytes:.0f} MB) "
        f"to 16kHz/16-bit/mono WAV with {offset_seconds}s silence prepended..."
    )

    ffmpeg_arguments = [
        "ffmpeg", "-y",
        "-i", str(audio_file_path),
        "-vn",
        "-af", f"adelay={delay_milliseconds}|{delay_milliseconds}",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(converted_path),
    ]

    _run_ffmpeg_conversion(ffmpeg_arguments, converted_path)

    converted_size_megabytes = _format_file_size_megabytes(converted_path)
    _log(
        f"Converted: {source_size_megabytes:.0f} MB -> {converted_size_megabytes:.0f} MB "
        f"(16kHz/16-bit/mono + {offset_seconds}s offset)"
    )

    return converted_path


def compress_audio_if_needed(
    audio_file_path: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Convert audio to 16kHz mono 16-bit WAV for WhisperX.
    Only converts files above 50 MB.
    """
    def _log(message: str) -> None:
        logger.info(message)
        if on_progress:
            on_progress(message)

    file_size_bytes = audio_file_path.stat().st_size
    source_size_megabytes = _format_file_size_megabytes(audio_file_path)

    if file_size_bytes < COMPRESSION_THRESHOLD_BYTES:
        _log(f"Audio is small ({source_size_megabytes:.0f} MB), skipping conversion")
        return audio_file_path

    converted_path = audio_file_path.with_suffix(".16k.wav")

    if converted_path.exists():
        _log(f"Using existing converted file ({_format_file_size_megabytes(converted_path):.0f} MB)")
        return converted_path

    _log(f"Audio is {source_size_megabytes:.0f} MB -- converting to 16kHz/16-bit/mono WAV...")

    ffmpeg_arguments = [
        "ffmpeg", "-y",
        "-i", str(audio_file_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(converted_path),
    ]

    _run_ffmpeg_conversion(ffmpeg_arguments, converted_path)

    converted_size_megabytes = _format_file_size_megabytes(converted_path)
    _log(f"Converted: {source_size_megabytes:.0f} MB -> {converted_size_megabytes:.0f} MB (16kHz/16-bit/mono)")

    return converted_path


def compress_audio_for_transkriptor(
    audio_file_path: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Convert audio to MP3 mono 128kbps 16kHz for Transkriptor upload.
    Minimizes upload size while preserving speech clarity.
    Caches as {stem}.transkriptor.mp3.
    """
    def _log(message: str) -> None:
        logger.info(message)
        if on_progress:
            on_progress(message)

    compressed_path = audio_file_path.parent / f"{audio_file_path.stem}.transkriptor.mp3"

    if compressed_path.exists():
        _log(f"Using cached Transkriptor audio ({_format_file_size_megabytes(compressed_path):.0f} MB)")
        return compressed_path

    source_size_megabytes = _format_file_size_megabytes(audio_file_path)
    _log(
        f"Compressing {audio_file_path.name} ({source_size_megabytes:.0f} MB) "
        f"to MP3 mono {TRANSKRIPTOR_BITRATE} {TRANSKRIPTOR_SAMPLE_RATE}Hz..."
    )

    ffmpeg_arguments = [
        "ffmpeg", "-y",
        "-i", str(audio_file_path),
        "-vn",
        "-ac", "1",
        "-ar", TRANSKRIPTOR_SAMPLE_RATE,
        "-ab", TRANSKRIPTOR_BITRATE,
        "-acodec", "libmp3lame",
        str(compressed_path),
    ]

    _run_ffmpeg_conversion(ffmpeg_arguments, compressed_path)

    compressed_size_megabytes = _format_file_size_megabytes(compressed_path)
    _log(
        f"Compressed: {source_size_megabytes:.0f} MB -> {compressed_size_megabytes:.0f} MB "
        f"(MP3 mono {TRANSKRIPTOR_BITRATE} {TRANSKRIPTOR_SAMPLE_RATE}Hz)"
    )

    return compressed_path
