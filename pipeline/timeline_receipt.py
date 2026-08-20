"""
pipeline/timeline_receipt.py

Records, per clip, how the published portrait video's timeline derives from
the base clip timeline: keyframe head padding from the stream-copy cut,
the portrait speed multiplier, and the silence-trim kept segments.

The receipt lets any later stage (subtitle burning, word-level captions)
map a time on the subtitle/wav timeline onto the final video timeline
without re-deriving the transform chain.

Timelines involved:
- subtitle timeline: zero = the requested segment start (SRT, wavs, words.json)
- base mp4 timeline: zero = the keyframe before the requested start
  (stream copy seeks back), so it leads the subtitle timeline by
  head_padding_seconds
- portrait timeline: base mp4 timeline divided by the speed multiplier
- trimmed timeline: portrait timeline with silence gaps removed
  (kept segments concatenated); the [square] variant shares it
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

RECEIPT_VERSION = 1

TIMELINE_DIRECTORY_NAME = ".cdata/timeline"


def build_receipt_file_path(clips_directory: Path, clip_base_name: str) -> Path:
    return clips_directory / TIMELINE_DIRECTORY_NAME / f"{clip_base_name}.timeline.json"


def load_timeline_receipt(clips_directory: Path, clip_base_name: str) -> dict:
    receipt_path = build_receipt_file_path(clips_directory, clip_base_name)
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    return {"version": RECEIPT_VERSION, "clipBase": clip_base_name}


def save_timeline_receipt(clips_directory: Path, receipt: dict) -> Path:
    receipt_path = build_receipt_file_path(clips_directory, receipt["clipBase"])
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt["updatedUtc"] = datetime.now(timezone.utc).isoformat()
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("Timeline receipt saved: %s", receipt_path.name)
    return receipt_path


def build_base_stage(
    requested_start_seconds: float,
    requested_end_seconds: float,
    mp4_duration_seconds: float,
) -> dict:
    """Describe the stream-copy cut of the base clip.

    head_padding = how much earlier the mp4 starts than the requested window
    (the keyframe seek-back). Subtitle timeline + head_padding = mp4 timeline.
    """
    requested_duration = requested_end_seconds - requested_start_seconds
    head_padding_seconds = max(0.0, mp4_duration_seconds - requested_duration)
    return {
        "requestedStartSec": round(requested_start_seconds, 3),
        "requestedEndSec": round(requested_end_seconds, 3),
        "mp4DurationSec": round(mp4_duration_seconds, 3),
        "headPaddingSec": round(head_padding_seconds, 3),
    }


def build_portrait_stage(
    source_duration_seconds: float,
    speed_multiplier: float,
) -> dict:
    return {
        "sourceDurationSec": round(source_duration_seconds, 3),
        "speedMultiplier": speed_multiplier,
    }


def build_trim_stage(kept_segments: Sequence[tuple[float, float]]) -> dict:
    """Kept (non-silent) segments on the portrait timeline, in order."""
    return {
        "keptSegments": [
            [round(start, 3), round(end, 3)] for start, end in kept_segments
        ],
    }


def map_subtitle_time_to_final(time_seconds: float, receipt: dict) -> Optional[float]:
    """Map a subtitle-timeline time onto the final published video timeline.

    Applies, in order: head padding (subtitle -> base mp4), portrait speed
    (base mp4 -> portrait), silence trim (portrait -> trimmed/square).
    Returns None when the time falls inside a trimmed-away silence gap.
    """
    base_stage = receipt.get("base")
    mapped = time_seconds
    if base_stage:
        mapped = time_seconds + base_stage["headPaddingSec"]

    portrait_stage = receipt.get("portrait")
    if portrait_stage:
        speed = portrait_stage["speedMultiplier"]
        if speed and speed != 1.0:
            mapped = mapped / speed

    trim_stage = receipt.get("trim")
    if trim_stage:
        mapped = _map_through_kept_segments(mapped, trim_stage["keptSegments"])
    return mapped


def _map_through_kept_segments(
    time_seconds: float,
    kept_segments: Sequence[Sequence[float]],
) -> Optional[float]:
    elapsed_kept = 0.0
    for segment_start, segment_end in kept_segments:
        if time_seconds < segment_start:
            return None  # inside a removed gap before this kept segment
        if time_seconds <= segment_end:
            return elapsed_kept + (time_seconds - segment_start)
        elapsed_kept += segment_end - segment_start
    return None  # past the last kept segment
