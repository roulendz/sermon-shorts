"""
Compare Transkriptor timestamps vs FFmpeg silence-detected speech regions.

Shows where Transkriptor entries span across silence gaps (covering time
where the other language is actually speaking).

Usage:
    python scripts/transkriptor_vs_silence_analysis.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.silence_detection import (
    detect_silence_regions,
    silence_regions_to_speech_regions,
    get_audio_duration_seconds,
)

BASE_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW")

RU_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A04.wav"
RU_CONTENT_JSON = BASE_DIR / "transkriptor_raw_A04_RU.json"

LV_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A03.wav"
LV_CONTENT_JSON = BASE_DIR / "transkriptor_raw_A03_LV.json"

ANALYSIS_LIMIT_SECONDS = 300  # first 5 min
NOISE_THRESHOLD_DB = -25


def load_content(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "body" in data and isinstance(data["body"], dict):
        return data["body"].get("content", [])
    return data.get("content", [])


def find_speech_regions_covering_entry(entry, speech_regions):
    """Find all speech regions that overlap with a Transkriptor entry."""
    entry_start = entry["StartTime"] / 1000.0
    entry_end = entry["EndTime"] / 1000.0
    covering = []
    for region_start, region_end in speech_regions:
        if region_start < entry_end and region_end > entry_start:
            covering.append((region_start, region_end))
    return covering


def find_silence_gaps_within_entry(entry, silence_regions):
    """Find silence gaps that fall INSIDE a Transkriptor entry's time range."""
    entry_start = entry["StartTime"] / 1000.0
    entry_end = entry["EndTime"] / 1000.0
    gaps_inside = []
    for silence_start, silence_end in silence_regions:
        # Silence must be fully or partially inside the entry
        gap_start = max(silence_start, entry_start)
        gap_end = min(silence_end, entry_end)
        if gap_end > gap_start + 0.1:  # at least 100ms gap
            gaps_inside.append((gap_start, gap_end, gap_end - gap_start))
    return gaps_inside


def format_ts(seconds):
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:05.2f}"


def analyze_track(label, audio_path, content_json_path, silence_regions, speech_regions):
    content = load_content(content_json_path)

    # Filter to analysis window
    content_in_window = [
        e for e in content
        if e["StartTime"] / 1000.0 < ANALYSIS_LIMIT_SECONDS
    ]

    print(f"\n{'=' * 90}")
    print(f"{label} TRACK ANALYSIS")
    print(f"{'=' * 90}")
    print(f"Transkriptor entries in window: {len(content_in_window)}")
    print(f"FFmpeg speech regions: {len(speech_regions)}")
    print(f"FFmpeg silence regions: {len(silence_regions)}")

    # ── Compare durations ───────────────────────────────────────────────
    total_transkriptor_duration = sum(
        (e["EndTime"] - e["StartTime"]) / 1000.0 for e in content_in_window
    )
    total_speech_duration = sum(
        end - start for start, end in speech_regions
        if start < ANALYSIS_LIMIT_SECONDS
    )

    print(f"\nTotal Transkriptor claimed duration: {total_transkriptor_duration:.1f}s")
    print(f"Total FFmpeg actual speech duration:  {total_speech_duration:.1f}s")
    print(f"Difference: {total_transkriptor_duration - total_speech_duration:.1f}s "
          f"(Transkriptor spans {(total_transkriptor_duration / total_speech_duration - 1) * 100:.0f}% more)")

    # ── Find entries that span multiple speech regions ──────────────────
    multi_region_entries = []
    entries_with_internal_gaps = []

    for entry in content_in_window:
        covering = find_speech_regions_covering_entry(entry, speech_regions)
        internal_gaps = find_silence_gaps_within_entry(entry, silence_regions)

        if len(covering) > 1:
            multi_region_entries.append((entry, covering, internal_gaps))
        if internal_gaps:
            entries_with_internal_gaps.append((entry, internal_gaps))

    print(f"\nEntries spanning MULTIPLE speech regions: {len(multi_region_entries)} / {len(content_in_window)}")
    print(f"Entries with internal silence gaps:       {len(entries_with_internal_gaps)} / {len(content_in_window)}")

    # ── Show worst offenders ────────────────────────────────────────────
    print(f"\n--- Entries spanning multiple speech regions (top 15) ---")
    multi_region_entries.sort(key=lambda x: len(x[1]), reverse=True)

    for entry, covering, internal_gaps in multi_region_entries[:15]:
        entry_start = entry["StartTime"] / 1000.0
        entry_end = entry["EndTime"] / 1000.0
        entry_duration = entry_end - entry_start
        total_gap_inside = sum(d for _, _, d in internal_gaps)
        text = entry["text"][:60]

        print(f"\n  [{format_ts(entry_start)} - {format_ts(entry_end)}] ({entry_duration:.1f}s) "
              f"spans {len(covering)} speech regions")
        print(f"  Text: \"{text}\"")
        print(f"  Silence inside: {total_gap_inside:.1f}s across {len(internal_gaps)} gaps")
        for gap_start, gap_end, gap_duration in internal_gaps:
            print(f"    GAP: {format_ts(gap_start)} - {format_ts(gap_end)} ({gap_duration:.1f}s)")
        for region_start, region_end in covering:
            print(f"    SPEECH: {format_ts(region_start)} - {format_ts(region_end)} ({region_end - region_start:.1f}s)")

    # ── Distribution of how many speech regions per entry ───────────────
    print(f"\n--- Distribution: speech regions per Transkriptor entry ---")
    from collections import Counter
    region_counts = Counter()
    for entry in content_in_window:
        covering = find_speech_regions_covering_entry(entry, speech_regions)
        region_counts[len(covering)] += 1

    for count in sorted(region_counts.keys()):
        bar = "█" * region_counts[count]
        print(f"  {count} region(s): {region_counts[count]:4d} entries  {bar}")

    return content_in_window, speech_regions, silence_regions


def main():
    print("=" * 90)
    print("TRANSKRIPTOR vs FFmpeg SILENCE ANALYSIS")
    print(f"Threshold: {NOISE_THRESHOLD_DB}dB, Window: {ANALYSIS_LIMIT_SECONDS}s")
    print("=" * 90)

    # Detect silences for both tracks
    print("\nRunning FFmpeg silence detection...")

    ru_duration = min(get_audio_duration_seconds(str(RU_AUDIO)), ANALYSIS_LIMIT_SECONDS)
    ru_silences = detect_silence_regions(str(RU_AUDIO), NOISE_THRESHOLD_DB, 1.0)
    ru_silences_in_window = [(s, e) for s, e in ru_silences if s < ANALYSIS_LIMIT_SECONDS]
    ru_speech = silence_regions_to_speech_regions(ru_silences_in_window, ru_duration)

    lv_duration = min(get_audio_duration_seconds(str(LV_AUDIO)), ANALYSIS_LIMIT_SECONDS)
    lv_silences = detect_silence_regions(str(LV_AUDIO), NOISE_THRESHOLD_DB, 1.0)
    lv_silences_in_window = [(s, e) for s, e in lv_silences if s < ANALYSIS_LIMIT_SECONDS]
    lv_speech = silence_regions_to_speech_regions(lv_silences_in_window, lv_duration)

    # Analyze each track
    analyze_track("RU", RU_AUDIO, RU_CONTENT_JSON, ru_silences_in_window, ru_speech)
    analyze_track("LV", LV_AUDIO, LV_CONTENT_JSON, lv_silences_in_window, lv_speech)

    # ── Cross-track: show where Transkriptor RU entry covers LV speech ──
    print(f"\n{'=' * 90}")
    print("CROSS-TRACK: RU Transkriptor entries overlapping LV speech regions")
    print(f"{'=' * 90}")

    ru_content = load_content(RU_CONTENT_JSON)
    ru_in_window = [e for e in ru_content if e["StartTime"] / 1000.0 < ANALYSIS_LIMIT_SECONDS]

    cross_overlaps = 0
    total_cross_overlap_seconds = 0

    for entry in ru_in_window:
        entry_start = entry["StartTime"] / 1000.0
        entry_end = entry["EndTime"] / 1000.0

        for lv_start, lv_end in lv_speech:
            overlap_start = max(entry_start, lv_start)
            overlap_end = min(entry_end, lv_end)
            if overlap_start < overlap_end:
                cross_overlaps += 1
                total_cross_overlap_seconds += overlap_end - overlap_start

    print(f"RU entries overlapping LV speech: {cross_overlaps} overlaps")
    print(f"Total cross-overlap duration: {total_cross_overlap_seconds:.1f}s")
    print(f"This is time Transkriptor claims RU is speaking, but actually LV is speaking.")

    print(f"\n{'=' * 90}")
    print("CONCLUSION")
    print(f"{'=' * 90}")
    print("Transkriptor merges speech across silence gaps into single sentences.")
    print("To fix: split Transkriptor entries at silence boundaries, then re-interleave.")
    print("Each split chunk gets proportional text (by word count or character count).")


if __name__ == "__main__":
    main()
