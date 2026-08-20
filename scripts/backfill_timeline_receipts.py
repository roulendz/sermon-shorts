"""
scripts/backfill_timeline_receipts.py

Backfill .cdata/timeline/<base>.timeline.json receipts for clips rendered
before receipt emission existed.

For each clip base in a Clips RAW folder:
- base stage: requested window from .data/shorts.json + probed mp4 duration
- portrait stage: speed derived from probed base/portrait durations
  (rendered before the tail-chop fix, portrait covers only part of the
  base mp4 — sourceDurationSec records what it actually covers)
- trim stage: silence detection re-run on the portrait mp4 with the same
  parameters the pipeline used (deterministic)

Usage:
    python scripts/backfill_timeline_receipts.py "U:\\<event folder>"
        [--min-silence 1.0]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.silence_remover import plan_silence_removal
from pipeline.timeline_receipt import (
    build_base_stage,
    build_portrait_stage,
    build_trim_stage,
    load_timeline_receipt,
    save_timeline_receipt,
)
from pipeline.video_probe import probe_duration_seconds

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_shorts_receipt(event_root: Path) -> dict[str, dict]:
    """Map clip base name -> shorts.json clip entry (for requested windows)."""
    shorts_path = event_root / ".data" / "shorts.json"
    if not shorts_path.exists():
        raise SystemExit(f"shorts.json not found: {shorts_path}")
    shorts = json.loads(shorts_path.read_text(encoding="utf-8"))
    clips_by_base = {}
    for clip in shorts.get("clips", []):
        base_name = Path(clip["mp4"]).stem
        clips_by_base[base_name] = clip
    return clips_by_base


def backfill_clip(
    clips_directory: Path,
    clip_base_name: str,
    shorts_clip: dict,
    minimum_silence_seconds: float,
) -> None:
    base_mp4 = clips_directory / f"{clip_base_name}.mp4"
    portrait_mp4 = clips_directory / f"{clip_base_name} [portrait].mp4"
    trimmed_mp4 = clips_directory / f"{clip_base_name} [portrait] [trimmed].mp4"

    receipt = load_timeline_receipt(clips_directory, clip_base_name)

    base_duration = probe_duration_seconds(base_mp4)
    receipt["base"] = build_base_stage(
        requested_start_seconds=shorts_clip["startSec"],
        requested_end_seconds=shorts_clip["endSec"],
        mp4_duration_seconds=base_duration,
    )

    if portrait_mp4.exists():
        portrait_duration = probe_duration_seconds(portrait_mp4)
        # Pre-fix portraits covered only requested-duration seconds of the base
        # mp4, not the whole file; record what the portrait actually spans and
        # derive the true speed from the two probed durations.
        requested_duration = shorts_clip["endSec"] - shorts_clip["startSec"]
        source_duration = min(base_duration, requested_duration)
        speed = round(source_duration / portrait_duration, 4) if portrait_duration else 1.0
        receipt["portrait"] = build_portrait_stage(source_duration, speed)

    if trimmed_mp4.exists() and portrait_mp4.exists():
        kept_segments = plan_silence_removal(
            portrait_mp4, minimum_duration_seconds=minimum_silence_seconds,
        )
        receipt["trim"] = build_trim_stage(kept_segments) if kept_segments else None
    else:
        receipt["trim"] = None

    save_timeline_receipt(clips_directory, receipt)
    logger.info("Backfilled: %s", clip_base_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_folder", type=Path)
    parser.add_argument("--min-silence", type=float, default=1.0)
    arguments = parser.parse_args()

    event_root = arguments.event_folder
    clips_directory = event_root / "Clips RAW"
    if not clips_directory.exists():
        raise SystemExit(f"Clips RAW not found in {event_root}")

    clips_by_base = load_shorts_receipt(event_root)
    for clip_base_name, shorts_clip in clips_by_base.items():
        if not (clips_directory / f"{clip_base_name}.mp4").exists():
            logger.warning("Skipping (mp4 missing): %s", clip_base_name)
            continue
        backfill_clip(
            clips_directory, clip_base_name, shorts_clip, arguments.min_silence,
        )


if __name__ == "__main__":
    main()
