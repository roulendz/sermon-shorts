"""
pipeline/silence_detection.py

FFmpeg-based silence detection for audio tracks.
Used to find precise speech boundaries for bilingual subtitle merging.

TUNING RESULTS (2026-03-17, tested on 5 min of sermon audio):
┌─────────────────────────────────────────────────────────────────────────┐
│ Approach              │ Overlaps │ Total    │ Notes                     │
├───────────────────────┼──────────┼──────────┼───────────────────────────┤
│ -20dB, 1.0s           │ 0        │ 0.00s    │ Aggressive, may clip      │
│                       │          │          │ quiet speech endings      │
│ -25dB, 1.0s           │ 1        │ 0.07s    │ ★ BEST — nearly perfect  │
│ -28dB, 1.0s           │ 6        │ 1.35s    │ Moderate bleed            │
│ -30dB, 1.0s           │ 11       │ 2.01s    │ Default FFmpeg-ish        │
│ -35dB, 1.0s           │ 15       │ 4.64s    │ Too sensitive             │
│ -40dB, 1.0s           │ 20       │ 6.90s    │ Way too sensitive         │
├───────────────────────┼──────────┼──────────┼───────────────────────────┤
│ -30dB, min_dur=0.3s   │ 8        │ 1.31s    │ Shorter min helps a bit   │
│ -30dB, min_dur=0.5s   │ 9        │ 1.43s    │                           │
│ -30dB, min_dur=2.0s   │ 13       │ 21.65s   │ Merges real pauses        │
├───────────────────────┼──────────┼──────────┼───────────────────────────┤
│ LV offset -0.3s       │ 5        │ 1.25s    │ Helps some, hurts others  │
│ LV offset +0.3s       │ 17       │ 5.43s    │ Makes it worse            │
├───────────────────────┼──────────┼──────────┼───────────────────────────┤
│ -25dB + split-middle   │ 0        │ 0.00s    │ ★ RECOMMENDED COMBO      │
│ -30dB + split-middle   │ 1        │ 0.04s    │ Fallback option           │
└─────────────────────────────────────────────────────────────────────────┘

If -25dB clips quiet speech, try:
  1. -28dB + split-middle (6 overlaps resolved → 0)
  2. -30dB + split-middle (11 overlaps → ~1 residual 0.04s)
  3. -30dB + min_dur=0.3s + split-middle (tighter segments)
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Recommended defaults from tuning
DEFAULT_NOISE_THRESHOLD_DB = -45
DEFAULT_MINIMUM_SILENCE_DURATION_SECONDS = 1.0


def detect_silence_regions(
    audio_file_path: str,
    noise_threshold_db: int = DEFAULT_NOISE_THRESHOLD_DB,
    minimum_silence_duration_seconds: float = DEFAULT_MINIMUM_SILENCE_DURATION_SECONDS,
    analysis_limit_seconds: float = 0,
    on_progress: Optional[Callable[[str], None]] = None,
) -> list[tuple[float, float]]:
    """
    Run FFmpeg silencedetect on an audio file.
    Returns list of (silence_start, silence_end) tuples in seconds.
    """
    def _log(message: str) -> None:
        logger.info(message)
        if on_progress:
            on_progress(message)

    command = ["ffmpeg"]
    if analysis_limit_seconds > 0:
        command += ["-t", str(analysis_limit_seconds)]
    command += [
        "-i", str(audio_file_path),
        "-af", f"silencedetect=noise={noise_threshold_db}dB:d={minimum_silence_duration_seconds}",
        "-f", "null", "-",
    ]

    logger.debug("FFmpeg command: %s", " ".join(command))
    _log(f"Running silence detection ({noise_threshold_db}dB, {minimum_silence_duration_seconds}s min)...")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")

    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", result.stderr)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", result.stderr)]

    silence_regions = list(zip(starts[:len(ends)], ends))

    # Handle trailing silence (start without matching end)
    if len(starts) > len(ends):
        fallback_end = analysis_limit_seconds if analysis_limit_seconds > 0 else starts[-1] + 60
        silence_regions.append((starts[-1], fallback_end))

    _log(f"Found {len(silence_regions)} silence regions")
    return silence_regions


def silence_regions_to_speech_regions(
    silence_regions: list[tuple[float, float]],
    total_duration_seconds: float,
) -> list[tuple[float, float]]:
    """
    Invert silence regions to get speech regions.
    Returns list of (speech_start, speech_end) tuples.
    """
    speech_regions = []
    previous_end = 0.0

    for silence_start, silence_end in silence_regions:
        if silence_start > previous_end + 0.05:
            speech_regions.append((previous_end, silence_start))
        previous_end = silence_end

    if previous_end < total_duration_seconds - 0.05:
        speech_regions.append((previous_end, total_duration_seconds))

    return speech_regions


def get_audio_duration_seconds(audio_file_path: str) -> float:
    """Get duration of audio file using FFprobe."""
    from pipeline.video_probe import probe_duration_seconds
    return probe_duration_seconds(Path(audio_file_path))
