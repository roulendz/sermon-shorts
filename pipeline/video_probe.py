"""
pipeline/video_probe.py

Shared FFprobe queries for video metadata.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def probe_duration_seconds(video_path: Path) -> float:
    """Get video/audio duration in seconds via ffprobe. Returns 0.0 on failure."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        duration = float(result.stdout.strip())
        logger.debug("Probed duration for %s: %.2fs", video_path.name, duration)
        return duration
    except ValueError:
        logger.warning("Failed to probe duration for %s", video_path)
        return 0.0
