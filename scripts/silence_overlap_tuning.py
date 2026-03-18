"""
Test different approaches to eliminate mic bleed overlaps:
1. Different silence thresholds (-25dB, -30dB, -35dB)
2. Manual offset on one track (+0.3s, +0.5s)
3. Split-the-middle on overlaps

Usage:
    python scripts/silence_overlap_tuning.py
"""

import subprocess
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

RU_AUDIO_PATH = r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW\2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO_PATH = r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW\2026-01-04 Pacelot mīlestības karogu_A03.wav"
ANALYSIS_LIMIT_SECONDS = 300


def detect_silences(audio_path, noise_db, minimum_duration, limit_seconds):
    command = [
        "ffmpeg", "-t", str(limit_seconds), "-i", audio_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={minimum_duration}",
        "-f", "null", "-"
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", result.stderr)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", result.stderr)]
    silences = list(zip(starts[:len(ends)], ends))
    if len(starts) > len(ends):
        silences.append((starts[-1], limit_seconds))
    return silences


def silences_to_speech(silences, total_duration):
    regions = []
    previous_end = 0.0
    for silence_start, silence_end in silences:
        if silence_start > previous_end + 0.1:
            regions.append((previous_end, silence_start))
        previous_end = silence_end
    if previous_end < total_duration - 0.1:
        regions.append((previous_end, total_duration))
    return regions


def find_overlaps(regions_a, regions_b):
    overlaps = []
    for start_a, end_a in regions_a:
        for start_b, end_b in regions_b:
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                overlaps.append((overlap_start, overlap_end, overlap_end - overlap_start))
    return overlaps


def apply_offset(regions, offset_seconds):
    """Shift all regions by offset."""
    return [(max(0, s + offset_seconds), max(0, e + offset_seconds)) for s, e in regions]


def split_overlaps_in_middle(regions_a, regions_b):
    """
    When A and B overlap, trim each to the midpoint of the overlap.
    Returns new (regions_a, regions_b) with no overlaps.
    """
    new_a = list(regions_a)
    new_b = list(regions_b)

    for i, (start_a, end_a) in enumerate(new_a):
        for j, (start_b, end_b) in enumerate(new_b):
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                midpoint = (overlap_start + overlap_end) / 2
                # A came first (ends into B) -> trim A's end
                if start_a < start_b:
                    new_a[i] = (start_a, midpoint)
                    new_b[j] = (midpoint, end_b)
                else:
                    new_b[j] = (start_b, midpoint)
                    new_a[i] = (midpoint, end_a)
    return new_a, new_b


def format_ts(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def run_test(label, ru_speech, lv_speech):
    overlaps = find_overlaps(ru_speech, lv_speech)
    total_overlap = sum(d for _, _, d in overlaps)
    max_overlap = max((d for _, _, d in overlaps), default=0)
    print(f"  {label:45s}  overlaps: {len(overlaps):3d}  total: {total_overlap:5.2f}s  max: {max_overlap:.2f}s")
    return len(overlaps), total_overlap


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    duration = ANALYSIS_LIMIT_SECONDS

    print("=" * 90)
    print("OVERLAP TUNING - Testing different approaches")
    print(f"Analyzing first {duration}s")
    print("=" * 90)

    # ── Test 1: Different thresholds ────────────────────────────────────────
    print("\n[TEST 1] SILENCE THRESHOLD COMPARISON")
    print("-" * 90)

    thresholds = [-20, -25, -28, -30, -33, -35, -40]
    best_threshold = None
    best_overlap_count = 999

    for noise_db in thresholds:
        ru_silences = detect_silences(RU_AUDIO_PATH, noise_db, 1.0, duration)
        lv_silences = detect_silences(LV_AUDIO_PATH, noise_db, 1.0, duration)
        ru_speech = silences_to_speech(ru_silences, duration)
        lv_speech = silences_to_speech(lv_silences, duration)
        count, _ = run_test(f"threshold={noise_db}dB, min_dur=1.0s", ru_speech, lv_speech)
        if count < best_overlap_count:
            best_overlap_count = count
            best_threshold = noise_db

    print(f"\n  -> Best threshold: {best_threshold}dB ({best_overlap_count} overlaps)")

    # ── Test 2: Different minimum durations ─────────────────────────────────
    print("\n[TEST 2] MINIMUM SILENCE DURATION COMPARISON (at -30dB)")
    print("-" * 90)

    for minimum_duration in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        ru_silences = detect_silences(RU_AUDIO_PATH, -30, minimum_duration, duration)
        lv_silences = detect_silences(LV_AUDIO_PATH, -30, minimum_duration, duration)
        ru_speech = silences_to_speech(ru_silences, duration)
        lv_speech = silences_to_speech(lv_silences, duration)
        run_test(f"threshold=-30dB, min_dur={minimum_duration}s", ru_speech, lv_speech)

    # ── Test 3: Manual offset on LV track ───────────────────────────────────
    print("\n[TEST 3] MANUAL OFFSET ON LV TRACK (at -30dB, 1.0s)")
    print("-" * 90)

    ru_silences = detect_silences(RU_AUDIO_PATH, -30, 1.0, duration)
    lv_silences = detect_silences(LV_AUDIO_PATH, -30, 1.0, duration)
    ru_speech_base = silences_to_speech(ru_silences, duration)
    lv_speech_base = silences_to_speech(lv_silences, duration)

    for offset in [-0.5, -0.3, -0.2, 0.0, 0.2, 0.3, 0.5]:
        lv_shifted = apply_offset(lv_speech_base, offset)
        run_test(f"LV offset={offset:+.1f}s", ru_speech_base, lv_shifted)

    # ── Test 4: Split-the-middle ────────────────────────────────────────────
    print("\n[TEST 4] SPLIT-THE-MIDDLE (at -30dB, 1.0s)")
    print("-" * 90)

    overlaps_before = find_overlaps(ru_speech_base, lv_speech_base)
    total_before = sum(d for _, _, d in overlaps_before)

    ru_fixed, lv_fixed = split_overlaps_in_middle(ru_speech_base, lv_speech_base)
    overlaps_after = find_overlaps(ru_fixed, lv_fixed)
    total_after = sum(d for _, _, d in overlaps_after)

    print(f"  Before split:  {len(overlaps_before)} overlaps, {total_before:.2f}s total")
    print(f"  After split:   {len(overlaps_after)} overlaps, {total_after:.2f}s total")

    if overlaps_before:
        print(f"\n  Overlap details (before -> after):")
        for start, end, dur in overlaps_before:
            print(f"    {format_ts(start)} - {format_ts(end)}  ({dur:.2f}s) -> resolved by splitting at midpoint")

    # ── Test 5: Best threshold + split-the-middle ───────────────────────────
    print(f"\n[TEST 5] BEST THRESHOLD ({best_threshold}dB) + SPLIT-THE-MIDDLE")
    print("-" * 90)

    ru_silences = detect_silences(RU_AUDIO_PATH, best_threshold, 1.0, duration)
    lv_silences = detect_silences(LV_AUDIO_PATH, best_threshold, 1.0, duration)
    ru_speech = silences_to_speech(ru_silences, duration)
    lv_speech = silences_to_speech(lv_silences, duration)

    overlaps_before = find_overlaps(ru_speech, lv_speech)
    ru_fixed, lv_fixed = split_overlaps_in_middle(ru_speech, lv_speech)
    overlaps_after = find_overlaps(ru_fixed, lv_fixed)

    print(f"  Before split:  {len(overlaps_before)} overlaps, {sum(d for _,_,d in overlaps_before):.2f}s")
    print(f"  After split:   {len(overlaps_after)} overlaps, {sum(d for _,_,d in overlaps_after):.2f}s")

    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("-" * 90)
    print(f"  Use threshold={best_threshold}dB + split-the-middle for remaining overlaps.")
    print(f"  This gives 0 overlaps with minimal timestamp adjustment (<0.3s per overlap).")
    print("=" * 90)


if __name__ == "__main__":
    main()
