"""
pipeline/transcription_runner.py

Orchestrates the transcription step: submits audio to WhisperX or Transkriptor
via the API client, applies the audio-to-video offset, and saves the aligned
SRT file with service metadata appended.

Bridges api/whisperx_client.py, api/transkriptor_client.py, and
pipeline/subtitle_parser.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional, Union

from api.whisperx_client import WhisperXClient
from api.transkriptor_client import TranskriptorClient
from pipeline.subtitle_parser import (
    parse_subtitle_text,
    shift_subtitle_timestamps,
    save_subtitles_to_file,
    serialize_subtitles_to_srt_text,
)

logger = logging.getLogger(__name__)

SERVICE_WHISPERX = "wsp"
SERVICE_TRANSKRIPTOR = "tra"

SERVICE_DISPLAY_NAMES = {
    SERVICE_WHISPERX: "WhisperX",
    SERVICE_TRANSKRIPTOR: "Transkriptor",
}


def transcribe_and_save_aligned_subtitles(
    audio_file_path: Path,
    output_subtitle_file_path: Path,
    whisperx_client: WhisperXClient,
    audio_to_video_offset_seconds: float,
    language_code: str = "en",
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Transcribe the audio file via WhisperX, shift timestamps to align with
    the video, and save the result as an SRT file.

    Returns the path to the saved SRT file.
    """
    def _log(message: str) -> None:
        logger.info(message)
        if on_progress:
            on_progress(message)

    _log(f"Starting transcription for: {audio_file_path.name}")

    raw_srt_text = whisperx_client.transcribe_audio_file(
        audio_file_path=audio_file_path,
        language_code=language_code,
        on_progress=on_progress,
    )

    subtitles = parse_subtitle_text(raw_srt_text)
    _log(f"Received {len(subtitles)} subtitle entries from WhisperX")

    if audio_to_video_offset_seconds != 0.0:
        offset = timedelta(seconds=audio_to_video_offset_seconds)
        _log(f"Shifting subtitle timestamps by {audio_to_video_offset_seconds:.3f}s")
        subtitles = shift_subtitle_timestamps(subtitles, offset)
        _log(f"Subtitles after offset shift: {len(subtitles)} entries remain")

    output_subtitle_file_path.parent.mkdir(parents=True, exist_ok=True)
    _save_subtitles_with_metadata(subtitles, output_subtitle_file_path, SERVICE_WHISPERX)
    _log(f"Aligned subtitles saved to: {output_subtitle_file_path}")

    return output_subtitle_file_path


def transcribe_with_transkriptor_and_save_aligned_subtitles(
    audio_file_path: Path,
    output_subtitle_file_path: Path,
    transkriptor_client: TranskriptorClient,
    audio_to_video_offset_seconds: float,
    language_locale: str = "lv-LV",
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Transcribe the audio file via Transkriptor, shift timestamps to align
    with the video, and save the result as an SRT file.

    Returns the path to the saved SRT file.
    """
    def _log(message: str) -> None:
        logger.info(message)
        if on_progress:
            on_progress(message)

    _log(f"Starting Transkriptor transcription for: {audio_file_path.name}")

    raw_srt_text = transkriptor_client.transcribe_audio_file(
        audio_file_path=audio_file_path,
        language_code=language_locale,
        on_progress=on_progress,
    )

    subtitles = parse_subtitle_text(raw_srt_text)
    _log(f"Received {len(subtitles)} subtitle entries from Transkriptor")

    if audio_to_video_offset_seconds != 0.0:
        offset = timedelta(seconds=audio_to_video_offset_seconds)
        _log(f"Shifting subtitle timestamps by {audio_to_video_offset_seconds:.3f}s")
        subtitles = shift_subtitle_timestamps(subtitles, offset)
        _log(f"Subtitles after offset shift: {len(subtitles)} entries remain")

    output_subtitle_file_path.parent.mkdir(parents=True, exist_ok=True)
    _save_subtitles_with_metadata(subtitles, output_subtitle_file_path, SERVICE_TRANSKRIPTOR)
    _log(f"Aligned subtitles saved to: {output_subtitle_file_path}")

    return output_subtitle_file_path


def build_transcription_output_path(
    video_file_path: Path,
    audio_file_path: Path,
    service_code: str,
) -> Path:
    """
    Build output path: {video_root}/transcriptions/{audio_name}_{service}_{timestamp}.srt
    """
    video_directory = video_file_path.parent
    audio_name = audio_file_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{audio_name}_{service_code}_{timestamp}.srt"
    return video_directory / "transcriptions" / filename


def _save_subtitles_with_metadata(
    subtitles,
    output_file_path: Path,
    service_code: str,
) -> None:
    """Save subtitles to file with transcription service metadata appended."""
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    srt_text = serialize_subtitles_to_srt_text(subtitles)

    service_name = SERVICE_DISPLAY_NAMES.get(service_code, service_code)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    metadata_line = f"\n# Transcription service: {service_name} | Date: {timestamp}\n"

    output_file_path.write_text(
        srt_text + metadata_line,
        encoding="utf-8",
    )
