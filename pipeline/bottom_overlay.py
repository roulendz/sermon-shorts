"""
pipeline/bottom_overlay.py

Adds a solid-color rectangle at the bottom of a portrait video using FFmpeg drawbox.
Color is sampled from a configurable region of the first frame, with automatic
fallback to the opposite side when the primary sample hits a non-wall area.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from pipeline.video_probe import probe_duration_seconds

logger = logging.getLogger(__name__)

DEFAULT_OVERLAY_HEIGHT = 695
DEFAULT_COLOR_SAMPLE_SIZE = 25
DEFAULT_COLOR_SAMPLE_OFFSET_X = 100
DEFAULT_COLOR_SAMPLE_OFFSET_Y = 1300
FALLBACK_COLOR_SAMPLE_OFFSET_X = 3600


class BottomOverlayError(Exception):
    pass


def read_first_frame(video_path: Path) -> np.ndarray:
    """Read and return the first frame of a video as BGR numpy array."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise BottomOverlayError(f"Cannot open video: {video_path}")

    success, frame = capture.read()
    capture.release()

    if not success:
        raise BottomOverlayError(f"Cannot read first frame from: {video_path}")

    return frame


def sample_color_from_frame(
    frame: np.ndarray,
    offset_x: int,
    offset_y: int,
    sample_size: int = DEFAULT_COLOR_SAMPLE_SIZE,
) -> np.ndarray:
    """Sample average BGR color from a square region of a frame."""
    frame_height, frame_width = frame.shape[:2]

    clamped_x = min(offset_x, frame_width - sample_size)
    clamped_y = min(offset_y, frame_height - sample_size)
    region_width = min(sample_size, frame_width - clamped_x)
    region_height = min(sample_size, frame_height - clamped_y)

    sample_region = frame[clamped_y:clamped_y + region_height, clamped_x:clamped_x + region_width]
    return sample_region.mean(axis=(0, 1)).astype(np.uint8)


def is_cool_toned(bgr_color: np.ndarray) -> bool:
    """Check if a BGR color is cool-toned (blue dominant over red). Wall = cool, podium/skin = warm."""
    blue_channel = int(bgr_color[0])
    red_channel = int(bgr_color[2])
    return blue_channel > red_channel


def sample_wall_color(
    video_path: Path,
    sample_size: int = DEFAULT_COLOR_SAMPLE_SIZE,
    primary_offset_x: int = DEFAULT_COLOR_SAMPLE_OFFSET_X,
    fallback_offset_x: int = FALLBACK_COLOR_SAMPLE_OFFSET_X,
    offset_y: int = DEFAULT_COLOR_SAMPLE_OFFSET_Y,
) -> np.ndarray:
    """
    Sample wall color from a video frame with automatic fallback.

    Tries the primary X position first. If the sampled color is warm-toned
    (pastor or podium blocking the wall), falls back to the opposite side.
    """
    frame = read_first_frame(video_path)

    primary_color = sample_color_from_frame(frame, primary_offset_x, offset_y, sample_size)

    if is_cool_toned(primary_color):
        logger.info(
            f"Primary sample at ({primary_offset_x},{offset_y}) "
            f"BGR=({primary_color[0]},{primary_color[1]},{primary_color[2]}) — wall color"
        )
        return primary_color

    fallback_color = sample_color_from_frame(frame, fallback_offset_x, offset_y, sample_size)
    logger.info(
        f"Primary sample at ({primary_offset_x},{offset_y}) "
        f"BGR=({primary_color[0]},{primary_color[1]},{primary_color[2]}) — warm, using fallback "
        f"at ({fallback_offset_x},{offset_y}) "
        f"BGR=({fallback_color[0]},{fallback_color[1]},{fallback_color[2]})"
    )
    return fallback_color


def bgr_to_hex_rgb(bgr_color: np.ndarray) -> str:
    """Convert BGR numpy array to hex RGB string like 'a1b2c3'."""
    return f"{bgr_color[2]:02x}{bgr_color[1]:02x}{bgr_color[0]:02x}"


def apply_bottom_overlay(
    source_video_path: Path,
    output_file_path: Path,
    landscape_video_path: Path,
    overlay_height: int = DEFAULT_OVERLAY_HEIGHT,
    color_sample_size: int = DEFAULT_COLOR_SAMPLE_SIZE,
    color_sample_offset_x: int = DEFAULT_COLOR_SAMPLE_OFFSET_X,
    color_sample_offset_y: int = DEFAULT_COLOR_SAMPLE_OFFSET_Y,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Path:
    """
    Add a solid-color rectangle at the bottom of a video using FFmpeg drawbox.
    Color is always sampled from the landscape original, never from portrait/trimmed.
    Single-pass re-encode — no Python frame loop.
    """
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    overlay_color_bgr = sample_wall_color(
        video_path=landscape_video_path,
        sample_size=color_sample_size,
        primary_offset_x=color_sample_offset_x,
        offset_y=color_sample_offset_y,
    )
    hex_color = bgr_to_hex_rgb(overlay_color_bgr)

    ffmpeg_arguments = _build_drawbox_command(
        source_video_path=source_video_path,
        output_file_path=output_file_path,
        overlay_height=overlay_height,
        hex_color=hex_color,
    )

    logger.info(
        "Bottom overlay: %s → %s (color=#%s, height=%dpx)",
        source_video_path.name, output_file_path.name, hex_color, overlay_height,
    )
    logger.debug("FFmpeg command: %s", " ".join(ffmpeg_arguments))

    process = subprocess.Popen(
        ffmpeg_arguments,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
    )

    _report_encoding_progress(process, source_video_path, on_progress)

    process.wait()

    if process.returncode != 0:
        raise BottomOverlayError(
            f"FFmpeg overlay encoding failed (exit code {process.returncode})"
        )

    if on_progress:
        on_progress(100.0)

    logger.info(f"Bottom overlay saved: {output_file_path}")
    return output_file_path


def _build_drawbox_command(
    source_video_path: Path,
    output_file_path: Path,
    overlay_height: int,
    hex_color: str,
) -> list[str]:
    """Build FFmpeg command with drawbox filter for bottom overlay."""
    return [
        "ffmpeg", "-y",
        "-i", str(source_video_path),
        "-vf", f"drawbox=x=0:y=ih-{overlay_height}:w=iw:h={overlay_height}:color=0x{hex_color}:t=fill",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.2",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_file_path),
    ]


def _report_encoding_progress(
    process: subprocess.Popen,
    source_video_path: Path,
    on_progress: Optional[Callable[[float], None]],
) -> None:
    """Parse FFmpeg stderr for time= updates and report progress percentage."""
    if not on_progress:
        process.stderr.read()
        return

    duration_seconds = probe_duration_seconds(source_video_path)
    progress_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

    for line in process.stderr:
        if duration_seconds <= 0:
            continue
        match = progress_pattern.search(line)
        if not match:
            continue
        elapsed = int(match.group(1)) * 3600 + int(match.group(2)) * 60 + float(match.group(3))
        on_progress(min(99.0, (elapsed / duration_seconds) * 100))


def build_square_output_path(source_video_path: Path) -> Path:
    """Append [square] tag to the filename."""
    stem = source_video_path.stem
    return source_video_path.parent / f"{stem} [square].mp4"


_PIPELINE_TAG_PATTERN = re.compile(
    r"\s*\[(?:portrait|debug|trimmed|square|portrait-[\d.]+x)\]"
)


def derive_landscape_clip_path(tagged_video_path: Path) -> Path:
    """Strip pipeline tags from filename to get the original landscape clip path."""
    clean_stem = _PIPELINE_TAG_PATTERN.sub("", tagged_video_path.stem)
    return tagged_video_path.parent / f"{clean_stem}{tagged_video_path.suffix}"
