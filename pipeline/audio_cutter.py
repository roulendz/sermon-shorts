"""
pipeline/audio_cutter.py

Cuts audio segments from offset-prepared audio files.
Each segment is saved as a separate WAV file alongside its video clip.

Uses the same FFmpeg cut pattern as video_cutter but for audio-only sources.
"""

from __future__ import annotations

import logging
from pathlib import Path

from models.video_segment import VideoSegment
from pipeline.video_cutter import (
    build_clip_base_filename,
    build_ffmpeg_cut_arguments,
    run_ffmpeg_command,
)

logger = logging.getLogger(__name__)


def cut_audio_segment(
    source_audio_path: Path,
    segment: VideoSegment,
    output_file_path: Path,
) -> Path:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_arguments = build_ffmpeg_cut_arguments(
        source_video_path=source_audio_path,
        start_timestamp=segment.ffmpeg_start_timestamp,
        end_timestamp=segment.ffmpeg_end_timestamp,
        output_file_path=output_file_path,
    )

    logger.info(
        f"Cutting audio segment {segment.index}: "
        f"{segment.ffmpeg_start_timestamp} → {segment.ffmpeg_end_timestamp} "
        f"({segment.duration_seconds:.1f}s)"
    )

    run_ffmpeg_command(ffmpeg_arguments)
    logger.info(f"Audio segment {segment.index} saved: {output_file_path}")
    return output_file_path


def build_output_audio_file_path(
    output_directory: Path,
    segment: VideoSegment,
    language_code: str,
    video_file_path: Path | None = None,
) -> Path:
    base = build_clip_base_filename(segment, video_file_path)
    return output_directory / f"{base} [{language_code}].wav"
