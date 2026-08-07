"""
pipeline/audio_compressor.py

Audio format conversion for transcription services.
- prepare_audio_with_offset: 16kHz mono WAV with silence prepend (source of truth)
- compress_audio_for_transkriptor: MP3 128kbps format conversion (no offset, input already offset)
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

WHISPERX_CODEC_ARGUMENTS = ["-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1"]
WHISPERX_FORMAT_LABEL = "16kHz/16-bit/mono WAV"

ORIGINAL_QUALITY_CODEC_ARGUMENTS = ["-acodec", "pcm_s16le"]
ORIGINAL_QUALITY_FORMAT_LABEL = "original-quality WAV"

TRANSKRIPTOR_CODEC_ARGUMENTS = [
    "-ac", "1",
    "-ar", TRANSKRIPTOR_SAMPLE_RATE,
    "-ab", TRANSKRIPTOR_BITRATE,
    "-acodec", "libmp3lame",
]
TRANSKRIPTOR_FORMAT_LABEL = f"MP3 mono {TRANSKRIPTOR_BITRATE} {TRANSKRIPTOR_SAMPLE_RATE}Hz"


def _build_adelay_filter_value(offset_seconds: float) -> str:
    delay_milliseconds = int(offset_seconds * 1000)
    return f"adelay={delay_milliseconds}|{delay_milliseconds}"


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


def _convert_audio(
    audio_file_path: Path,
    output_suffix: str,
    codec_arguments: list[str],
    format_label: str,
    offset_seconds: float = 0.0,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    def _log(message: str) -> None:
        logger.info(message)
        if on_progress:
            on_progress(message)

    output_path = audio_file_path.parent / f"{audio_file_path.stem}{output_suffix}"

    if output_path.exists():
        _log(f"Using cached audio ({_format_file_size_megabytes(output_path):.0f} MB)")
        return output_path

    source_size_megabytes = _format_file_size_megabytes(audio_file_path)
    has_offset = offset_seconds > 0.0
    offset_label = f" + {offset_seconds}s offset" if has_offset else ""

    _log(
        f"Converting {audio_file_path.name} ({source_size_megabytes:.0f} MB) "
        f"to {format_label}{offset_label}..."
    )

    adelay_filter_arguments = ["-af", _build_adelay_filter_value(offset_seconds)] if has_offset else []

    ffmpeg_arguments = [
        "ffmpeg", "-y",
        "-i", str(audio_file_path),
        "-vn",
        *adelay_filter_arguments,
        *codec_arguments,
        str(output_path),
    ]

    _run_ffmpeg_conversion(ffmpeg_arguments, output_path)

    output_size_megabytes = _format_file_size_megabytes(output_path)
    _log(
        f"Converted: {source_size_megabytes:.0f} MB -> {output_size_megabytes:.0f} MB "
        f"({format_label}{offset_label})"
    )

    return output_path


def prepare_audio_with_offset(
    audio_file_path: Path,
    offset_seconds: float = DEFAULT_AUDIO_OFFSET_SECONDS,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    return _convert_audio(
        audio_file_path,
        output_suffix=".offset.16k.wav",
        codec_arguments=WHISPERX_CODEC_ARGUMENTS,
        format_label=WHISPERX_FORMAT_LABEL,
        offset_seconds=offset_seconds,
        on_progress=on_progress,
    )


def prepare_original_audio_with_offset(
    audio_file_path: Path,
    offset_seconds: float = DEFAULT_AUDIO_OFFSET_SECONDS,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    return _convert_audio(
        audio_file_path,
        output_suffix=".offset.wav",
        codec_arguments=ORIGINAL_QUALITY_CODEC_ARGUMENTS,
        format_label=ORIGINAL_QUALITY_FORMAT_LABEL,
        offset_seconds=offset_seconds,
        on_progress=on_progress,
    )


def compress_audio_for_transkriptor(
    audio_file_path: Path,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    return _convert_audio(
        audio_file_path,
        output_suffix=".transkriptor.mp3",
        codec_arguments=TRANSKRIPTOR_CODEC_ARGUMENTS,
        format_label=TRANSKRIPTOR_FORMAT_LABEL,
        on_progress=on_progress,
    )


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
