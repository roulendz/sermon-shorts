"""
Merge RU + LV subtitles with silence-based splitting, then validate:
- RU text only appears during RU speech (no silence)
- LV text only appears during LV speech (no silence)
- No overlapping timestamps between RU and LV entries

Usage:
    python scripts/merge_and_validate_poc.py
"""

import sys
from pathlib import Path
from datetime import timedelta

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

import srt
from pipeline.bilingual_subtitle_merger import (
    merge_bilingual_subtitles,
    load_transkriptor_content_from_json,
)
from pipeline.silence_detection import (
    detect_silence_regions,
    silence_regions_to_speech_regions,
    get_audio_duration_seconds,
)

BASE_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW")

RU_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A03.wav"
RU_CONTENT_JSON = BASE_DIR / "transkriptor_raw_A04_RU.json"
LV_CONTENT_JSON = BASE_DIR / "transkriptor_raw_A03_LV.json"
from datetime import datetime
_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_SRT = BASE_DIR.parent / "transcriptions" / f"{_timestamp}_merged_RU_LV.srt"

AUDIO_TO_VIDEO_OFFSET_SECONDS = 9.13
NOISE_THRESHOLD_DB = -45
VALIDATION_WINDOW_SECONDS = 300  # validate first 5 min


def format_ts(seconds):
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:05.2f}"


def main():
    # ── Step 1: Merge ───────────────────────────────────────────────────
    print("=" * 80)
    print("STEP 1: MERGE WITH SILENCE-BASED SPLITTING")
    print("=" * 80)

    ru_content = load_transkriptor_content_from_json(RU_CONTENT_JSON)
    lv_content = load_transkriptor_content_from_json(LV_CONTENT_JSON)
    print(f"Loaded: RU {len(ru_content)} entries, LV {len(lv_content)} entries\n")

    output_path = merge_bilingual_subtitles(
        primary_audio_path=RU_AUDIO,
        primary_language_tag="RU",
        primary_content=ru_content,
        secondary_audio_path=LV_AUDIO,
        secondary_language_tag="LV",
        secondary_content=lv_content,
        output_srt_path=OUTPUT_SRT,
        audio_to_video_offset_seconds=AUDIO_TO_VIDEO_OFFSET_SECONDS,
        noise_threshold_db=NOISE_THRESHOLD_DB,
        on_progress=lambda message: print(f"  {message}"),
    )

    # ── Step 2: Load and preview ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("STEP 2: PREVIEW (first 30 entries)")
    print("=" * 80)

    subtitles = list(srt.parse(output_path.read_text(encoding="utf-8")))
    for subtitle in subtitles[:30]:
        start_s = subtitle.start.total_seconds()
        end_s = subtitle.end.total_seconds()
        duration = end_s - start_s
        print(f"  {subtitle.index:4d}. [{format_ts(start_s)} - {format_ts(end_s)}] ({duration:4.1f}s) {subtitle.content[:70]}")

    print(f"\n  ... {len(subtitles)} total entries")

    # ── Step 3: Validate against speech regions ─────────────────────────
    print(f"\n{'=' * 80}")
    print("STEP 3: VALIDATION (first 5 min)")
    print("=" * 80)

    # Get speech regions for validation (no offset — raw audio time)
    ru_duration = get_audio_duration_seconds(str(RU_AUDIO))
    ru_silences = detect_silence_regions(str(RU_AUDIO), NOISE_THRESHOLD_DB, 1.0)
    ru_speech = silence_regions_to_speech_regions(ru_silences, ru_duration)

    lv_duration = get_audio_duration_seconds(str(LV_AUDIO))
    lv_silences = detect_silence_regions(str(LV_AUDIO), NOISE_THRESHOLD_DB, 1.0)
    lv_speech = silence_regions_to_speech_regions(lv_silences, lv_duration)

    # Filter subtitles to validation window (convert back to audio time)
    offset = AUDIO_TO_VIDEO_OFFSET_SECONDS
    window_end_video = VALIDATION_WINDOW_SECONDS + offset

    ru_subs = []
    lv_subs = []
    for subtitle in subtitles:
        video_start = subtitle.start.total_seconds()
        video_end = subtitle.end.total_seconds()
        if video_start > window_end_video:
            break
        audio_start = video_start - offset
        audio_end = video_end - offset
        if audio_start < 0:
            continue
        if "[RU]" in subtitle.content:
            ru_subs.append((audio_start, audio_end, subtitle.content))
        elif "[LV]" in subtitle.content:
            lv_subs.append((audio_start, audio_end, subtitle.content))

    print(f"\n  RU subtitles in window: {len(ru_subs)}")
    print(f"  LV subtitles in window: {len(lv_subs)}")

    # ── Validate: RU subs should fall within RU speech regions ──────────
    print(f"\n--- RU subtitle vs RU speech alignment ---")
    ru_misaligned = 0
    ru_total_outside = 0.0

    for sub_start, sub_end, text in ru_subs:
        if sub_start >= VALIDATION_WINDOW_SECONDS:
            continue
        # How much of this subtitle falls OUTSIDE any RU speech region?
        outside_seconds = _time_outside_regions(sub_start, sub_end, ru_speech)
        if outside_seconds > 0.15:  # tolerance: 150ms
            ru_misaligned += 1
            ru_total_outside += outside_seconds
            if ru_misaligned <= 10:
                print(f"  MISALIGNED: [{format_ts(sub_start)} - {format_ts(sub_end)}] "
                      f"{outside_seconds:.2f}s outside RU speech")
                print(f"             {text[:60]}")

    if ru_misaligned > 10:
        print(f"  ... and {ru_misaligned - 10} more")
    print(f"  Total: {ru_misaligned} misaligned RU subs, {ru_total_outside:.1f}s outside speech")

    # ── Validate: LV subs should fall within LV speech regions ──────────
    print(f"\n--- LV subtitle vs LV speech alignment ---")
    lv_misaligned = 0
    lv_total_outside = 0.0

    for sub_start, sub_end, text in lv_subs:
        if sub_start >= VALIDATION_WINDOW_SECONDS:
            continue
        outside_seconds = _time_outside_regions(sub_start, sub_end, lv_speech)
        if outside_seconds > 0.15:
            lv_misaligned += 1
            lv_total_outside += outside_seconds
            if lv_misaligned <= 10:
                print(f"  MISALIGNED: [{format_ts(sub_start)} - {format_ts(sub_end)}] "
                      f"{outside_seconds:.2f}s outside LV speech")
                print(f"             {text[:60]}")

    if lv_misaligned > 10:
        print(f"  ... and {lv_misaligned - 10} more")
    print(f"  Total: {lv_misaligned} misaligned LV subs, {lv_total_outside:.1f}s outside speech")

    # ── Validate: RU subs should NOT overlap with LV speech ─────────────
    print(f"\n--- Cross-language: RU subs during LV speech ---")
    cross_overlap_count = 0
    cross_overlap_total = 0.0

    for sub_start, sub_end, text in ru_subs:
        if sub_start >= VALIDATION_WINDOW_SECONDS:
            continue
        inside_lv = _time_inside_regions(sub_start, sub_end, lv_speech)
        if inside_lv > 0.15:
            cross_overlap_count += 1
            cross_overlap_total += inside_lv
            if cross_overlap_count <= 5:
                print(f"  CROSS: [{format_ts(sub_start)} - {format_ts(sub_end)}] "
                      f"{inside_lv:.2f}s overlaps LV speech")
                print(f"         {text[:60]}")

    if cross_overlap_count > 5:
        print(f"  ... and {cross_overlap_count - 5} more")
    print(f"  Total: {cross_overlap_count} RU subs overlapping LV speech, {cross_overlap_total:.1f}s")

    print(f"\n--- Cross-language: LV subs during RU speech ---")
    cross_overlap_count_lv = 0
    cross_overlap_total_lv = 0.0

    for sub_start, sub_end, text in lv_subs:
        if sub_start >= VALIDATION_WINDOW_SECONDS:
            continue
        inside_ru = _time_inside_regions(sub_start, sub_end, ru_speech)
        if inside_ru > 0.15:
            cross_overlap_count_lv += 1
            cross_overlap_total_lv += inside_ru
            if cross_overlap_count_lv <= 5:
                print(f"  CROSS: [{format_ts(sub_start)} - {format_ts(sub_end)}] "
                      f"{inside_ru:.2f}s overlaps RU speech")
                print(f"         {text[:60]}")

    if cross_overlap_count_lv > 5:
        print(f"  ... and {cross_overlap_count_lv - 5} more")
    print(f"  Total: {cross_overlap_count_lv} LV subs overlapping RU speech, {cross_overlap_total_lv:.1f}s")

    # ── Validate: No RU/LV subtitle timestamp overlaps ──────────────────
    print(f"\n--- Subtitle timestamp overlaps (RU vs LV) ---")
    timestamp_overlaps = 0
    for ru_start, ru_end, ru_text in ru_subs:
        for lv_start, lv_end, lv_text in lv_subs:
            overlap_start = max(ru_start, lv_start)
            overlap_end = min(ru_end, lv_end)
            if overlap_start < overlap_end - 0.05:
                timestamp_overlaps += 1
                if timestamp_overlaps <= 5:
                    print(f"  OVERLAP ({overlap_end - overlap_start:.2f}s):")
                    print(f"    RU [{format_ts(ru_start)}-{format_ts(ru_end)}]: {ru_text[:50]}")
                    print(f"    LV [{format_ts(lv_start)}-{format_ts(lv_end)}]: {lv_text[:50]}")

    if timestamp_overlaps > 5:
        print(f"  ... and {timestamp_overlaps - 5} more")
    print(f"  Total: {timestamp_overlaps} RU/LV subtitle timestamp overlaps")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"  RU subs outside RU speech:  {ru_misaligned:4d} ({ru_total_outside:.1f}s)")
    print(f"  LV subs outside LV speech:  {lv_misaligned:4d} ({lv_total_outside:.1f}s)")
    print(f"  RU subs during LV speech:   {cross_overlap_count:4d} ({cross_overlap_total:.1f}s)")
    print(f"  LV subs during RU speech:   {cross_overlap_count_lv:4d} ({cross_overlap_total_lv:.1f}s)")
    print(f"  RU/LV timestamp overlaps:   {timestamp_overlaps:4d}")

    total_issues = ru_misaligned + lv_misaligned + cross_overlap_count + cross_overlap_count_lv + timestamp_overlaps
    if total_issues == 0:
        print(f"\n  ✓ PERFECT — all subtitles align with their speech regions!")
    else:
        print(f"\n  {total_issues} issues found — may need threshold tuning")

    print(f"\nOutput: {output_path}")


def _time_outside_regions(start, end, regions):
    """Calculate how many seconds of [start, end] fall OUTSIDE any region."""
    covered = 0.0
    for region_start, region_end in regions:
        overlap_start = max(start, region_start)
        overlap_end = min(end, region_end)
        if overlap_start < overlap_end:
            covered += overlap_end - overlap_start
    total = end - start
    return max(0, total - covered)


def _time_inside_regions(start, end, regions):
    """Calculate how many seconds of [start, end] fall INSIDE any region."""
    inside = 0.0
    for region_start, region_end in regions:
        overlap_start = max(start, region_start)
        overlap_end = min(end, region_end)
        if overlap_start < overlap_end:
            inside += overlap_end - overlap_start
    return inside


if __name__ == "__main__":
    main()
