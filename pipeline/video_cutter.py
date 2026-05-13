"""
pipeline/video_cutter.py

Builds FFmpeg commands and cuts video segments from the original video file.
Each segment is saved as a separate video file with an accompanying SRT file.

Only responsible for FFmpeg execution — no business logic about which
segments to cut (that lives in segment_selector.py).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from models.video_segment import VideoSegment

logger = logging.getLogger(__name__)


class FfmpegNotFoundError(Exception):
    pass


class VideoSegmentCuttingError(Exception):
    pass


def cut_segment_from_video(
    source_video_path: Path,
    segment: VideoSegment,
    output_file_path: Path,
) -> Path:
    """
    Cut a single segment from the source video using FFmpeg.
    Uses stream copy (no re-encoding) for speed when possible.
    Returns the path to the created file.
    """
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_arguments = build_ffmpeg_cut_arguments(
        source_video_path=source_video_path,
        start_timestamp=segment.ffmpeg_start_timestamp,
        end_timestamp=segment.ffmpeg_end_timestamp,
        output_file_path=output_file_path,
    )

    logger.info(
        f"Cutting segment {segment.index}: "
        f"{segment.ffmpeg_start_timestamp} → {segment.ffmpeg_end_timestamp} "
        f"({segment.duration_seconds:.1f}s)"
    )
    logger.debug(f"FFmpeg command: {' '.join(ffmpeg_arguments)}")

    run_ffmpeg_command(ffmpeg_arguments)
    logger.info(f"Segment {segment.index} saved: {output_file_path}")
    return output_file_path


def build_ffmpeg_cut_arguments(
    source_video_path: Path,
    start_timestamp: str,
    end_timestamp: str,
    output_file_path: Path,
) -> list[str]:
    """
    Build the FFmpeg argument list for cutting a segment.
    Uses -ss before -i for fast seeking, -c copy for no re-encoding.
    Pure function — no side effects, fully testable.
    """
    return [
        "ffmpeg",
        "-y",                           # overwrite output without asking
        "-ss", start_timestamp,         # seek before input (fast)
        "-to", end_timestamp,
        "-i", str(source_video_path),
        "-c", "copy",                   # no re-encoding — fast, lossless
        "-avoid_negative_ts", "make_zero",
        str(output_file_path),
    ]


def run_ffmpeg_command(ffmpeg_arguments: list[str]) -> None:
    """
    Execute an FFmpeg command and raise a descriptive error if it fails.
    """
    logger.debug("FFmpeg command: %s", " ".join(ffmpeg_arguments))
    try:
        result = subprocess.run(
            ffmpeg_arguments,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError:
        raise FfmpegNotFoundError(
            "FFmpeg executable not found. "
            "Install it with: brew install ffmpeg (Mac) or winget install ffmpeg (Windows)"
        )

    if result.returncode != 0:
        logger.error("FFmpeg failed (exit %d): %s", result.returncode, result.stderr[-500:])
        raise VideoSegmentCuttingError(
            f"FFmpeg failed with exit code {result.returncode}.\n"
            f"stderr: {result.stderr[-2000:]}"
        )


def _sanitize_filename(name: str) -> str:
    """Remove characters that are invalid in Windows/Mac filenames."""
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        name = name.replace(ch, "")
    # Collapse multiple spaces
    return " ".join(name.split()).strip()


def _extract_date_prefix(video_filename: str) -> str:
    """Extract date prefix like '2026-01-04' from video filename."""
    import re
    match = re.match(r"(\d{4}-\d{2}-\d{2})", video_filename)
    return match.group(1) if match else ""


def _format_duration_label(duration_seconds: float) -> str:
    """Format duration as human-readable label: '1m30s', '45s', '2m05s'."""
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    if minutes > 0:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def build_output_video_file_path(
    output_directory: Path,
    segment: VideoSegment,
    video_file_path: Path | None = None,
) -> Path:
    date_prefix = ""
    if video_file_path:
        date_prefix = _extract_date_prefix(video_file_path.stem)

    title = _sanitize_filename(segment.suggested_title or f"Segment {segment.index}")
    duration_label = _format_duration_label(segment.duration_seconds)
    if date_prefix:
        filename = f"{date_prefix} {title} [{duration_label}].mp4"
    else:
        filename = f"{title} [{duration_label}].mp4"

    return output_directory / filename


def build_output_subtitle_file_path(
    output_directory: Path,
    segment: VideoSegment,
    video_file_path: Path | None = None,
) -> Path:
    date_prefix = ""
    if video_file_path:
        date_prefix = _extract_date_prefix(video_file_path.stem)

    title = _sanitize_filename(segment.suggested_title or f"Segment {segment.index}")
    duration_label = _format_duration_label(segment.duration_seconds)
    if date_prefix:
        filename = f"{date_prefix} {title} [{duration_label}].srt"
    else:
        filename = f"{title} [{duration_label}].srt"

    return output_directory / filename
