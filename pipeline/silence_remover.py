"""
pipeline/silence_remover.py

Detects and removes silent segments from video files using FFmpeg.
Two-pass approach: detect silence timestamps, then concat non-silent ranges.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

SILENCE_DETECT_NOISE_THRESHOLD = "-30dB"


class SilenceRemovalError(Exception):
    pass


def detect_silence_ranges(
    video_path: Path,
    minimum_duration_seconds: float = 1.0,
) -> list[tuple[float, float]]:
    """
    Run FFmpeg silencedetect and return list of (start, end) tuples
    for each silent period longer than minimum_duration_seconds.
    """
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-af", f"silencedetect=n={SILENCE_DETECT_NOISE_THRESHOLD}:d={minimum_duration_seconds}",
        "-f", "null",
        "-",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    stderr_output = result.stderr
    silence_ranges: list[tuple[float, float]] = []

    start_pattern = re.compile(r"silence_start: ([\d.]+)")
    end_pattern = re.compile(r"silence_end: ([\d.]+)")

    starts: list[float] = []
    for line in stderr_output.splitlines():
        start_match = start_pattern.search(line)
        if start_match:
            starts.append(float(start_match.group(1)))

        end_match = end_pattern.search(line)
        if end_match and starts:
            silence_end = float(end_match.group(1))
            silence_start = starts.pop(0)
            silence_ranges.append((silence_start, silence_end))

    return silence_ranges


def build_non_silent_segments(
    total_duration: float,
    silence_ranges: list[tuple[float, float]],
    padding_seconds: float = 0.05,
) -> list[tuple[float, float]]:
    """
    Invert silence ranges into non-silent segments.
    Adds small padding around cuts to avoid clipping speech edges.
    """
    if not silence_ranges:
        return [(0.0, total_duration)]

    segments: list[tuple[float, float]] = []
    current_position = 0.0

    for silence_start, silence_end in silence_ranges:
        segment_end = silence_start + padding_seconds
        if segment_end > current_position:
            segments.append((current_position, segment_end))
        current_position = silence_end - padding_seconds

    if current_position < total_duration:
        segments.append((current_position, total_duration))

    return segments


def remove_silence_from_video(
    source_video_path: Path,
    output_file_path: Path,
    minimum_duration_seconds: float = 1.0,
    on_progress: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Detect silent periods and produce a new video with silences removed.
    Returns output path. Raises SilenceRemovalError on failure.
    """
    if on_progress:
        on_progress("Detecting silence...")

    silence_ranges = detect_silence_ranges(
        source_video_path,
        minimum_duration_seconds=minimum_duration_seconds,
    )

    if not silence_ranges:
        if on_progress:
            on_progress("No silence detected, skipping")
        return source_video_path

    total_silence = sum(end - start for start, end in silence_ranges)
    if on_progress:
        on_progress(f"Found {len(silence_ranges)} silent gaps ({total_silence:.1f}s total)")

    duration_result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(source_video_path)],
        capture_output=True, text=True,
    )
    total_duration = float(duration_result.stdout.strip())

    non_silent_segments = build_non_silent_segments(total_duration, silence_ranges)

    if on_progress:
        kept_duration = sum(end - start for start, end in non_silent_segments)
        on_progress(f"Keeping {kept_duration:.1f}s of {total_duration:.1f}s")

    filter_parts: list[str] = []
    concat_inputs: list[str] = []

    for index, (start, end) in enumerate(non_silent_segments):
        filter_parts.append(
            f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}];"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}];"
        )
        concat_inputs.append(f"[v{index}][a{index}]")

    filter_complex = "".join(filter_parts)
    filter_complex += f"{''.join(concat_inputs)}concat=n={len(non_silent_segments)}:v=1:a=1[outv][outa]"

    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y",
        "-i", str(source_video_path),
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_file_path),
    ]

    process_result = subprocess.run(command, capture_output=True, text=True)

    if process_result.returncode != 0:
        raise SilenceRemovalError(
            f"FFmpeg silence removal failed:\n{process_result.stderr[-2000:]}"
        )

    if on_progress:
        on_progress("Silence removal complete")

    logger.info(f"Silence removed: {output_file_path}")
    return output_file_path


def build_silence_removed_output_path(video_path: Path) -> Path:
    """Generate output path for silence-removed video."""
    stem = video_path.stem
    return video_path.parent / f"{stem} [trimmed].mp4"
