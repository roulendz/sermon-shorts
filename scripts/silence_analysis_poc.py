"""
Proof of concept: FFmpeg silence detection on RU and LV audio tracks.

Compares silence/speech patterns to verify they're complementary
(when one speaks, the other is silent) and identifies discrepancies.

Usage:
    python scripts/silence_analysis_poc.py
"""

import subprocess
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ── Config ──────────────────────────────────────────────────────────────────
RU_AUDIO_PATH = r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW\2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO_PATH = r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW\2026-01-04 Pacelot mīlestības karogu_A03.wav"

SILENCE_NOISE_THRESHOLD_DB = -30  # dB below which is considered silence
SILENCE_MINIMUM_DURATION_SECONDS = 1.0  # minimum silence gap to detect

# Only analyze first N seconds (0 = full file)
ANALYSIS_LIMIT_SECONDS = 300  # first 5 minutes for quick POC


# ── FFmpeg silence detection ────────────────────────────────────────────────
def detect_silences(audio_path, noise_db, minimum_duration, limit_seconds=0):
    """
    Run FFmpeg silencedetect and return list of (silence_start, silence_end) tuples.
    """
    command = ["ffmpeg", "-i", audio_path]
    if limit_seconds > 0:
        command = ["ffmpeg", "-t", str(limit_seconds), "-i", audio_path]

    command += [
        "-af", f"silencedetect=noise={noise_db}dB:d={minimum_duration}",
        "-f", "null", "-"
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    stderr_output = result.stderr

    # Parse silence_start and silence_end from FFmpeg output
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", stderr_output)]
    ends = [float(m.group(1)) for m in re.finditer(r"silence_end:\s*([\d.]+)", stderr_output)]

    # Pair them up
    silences = []
    for i in range(min(len(starts), len(ends))):
        silences.append((starts[i], ends[i]))

    # Handle trailing silence (start without end)
    if len(starts) > len(ends):
        silences.append((starts[-1], limit_seconds if limit_seconds > 0 else starts[-1] + 60))

    return silences


def silences_to_speech_regions(silences, total_duration):
    """
    Invert silence regions to get speech regions.
    Returns list of (speech_start, speech_end) tuples.
    """
    speech_regions = []
    previous_end = 0.0

    for silence_start, silence_end in silences:
        if silence_start > previous_end + 0.1:  # at least 100ms of speech
            speech_regions.append((previous_end, silence_start))
        previous_end = silence_end

    # Trailing speech after last silence
    if previous_end < total_duration - 0.1:
        speech_regions.append((previous_end, total_duration))

    return speech_regions


def format_timestamp(seconds):
    """Format seconds as MM:SS.s"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:05.2f}"


# ── Overlap analysis ───────────────────────────────────────────────────────
def find_overlaps(regions_a, regions_b):
    """
    Find time ranges where both A and B have speech simultaneously.
    Returns list of (overlap_start, overlap_end, duration).
    """
    overlaps = []
    for start_a, end_a in regions_a:
        for start_b, end_b in regions_b:
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                overlaps.append((overlap_start, overlap_end, overlap_end - overlap_start))
    return overlaps


def find_gaps(regions_a, regions_b, total_duration):
    """
    Find time ranges where NEITHER A nor B has speech (both silent).
    """
    # Merge all speech into one timeline
    all_speech = sorted(regions_a + regions_b, key=lambda x: x[0])

    # Merge overlapping speech regions
    merged = []
    for start, end in all_speech:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Gaps are where merged speech doesn't cover
    gaps = []
    previous_end = 0.0
    for start, end in merged:
        if start > previous_end + 0.1:
            gaps.append((previous_end, start, start - previous_end))
        previous_end = end

    if previous_end < total_duration - 0.1:
        gaps.append((previous_end, total_duration, total_duration - previous_end))

    return gaps


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    total_duration = ANALYSIS_LIMIT_SECONDS

    print("=" * 80)
    print("SILENCE ANALYSIS - Proof of Concept")
    print(f"Noise threshold: {SILENCE_NOISE_THRESHOLD_DB} dB")
    print(f"Min silence duration: {SILENCE_MINIMUM_DURATION_SECONDS}s")
    print(f"Analyzing first {total_duration}s ({total_duration // 60} min)")
    print("=" * 80)

    # Detect silences
    print("\n[1/4] Detecting silences in RU track...")
    ru_silences = detect_silences(
        RU_AUDIO_PATH, SILENCE_NOISE_THRESHOLD_DB,
        SILENCE_MINIMUM_DURATION_SECONDS, total_duration,
    )
    print(f"  Found {len(ru_silences)} silence regions")

    print("\n[2/4] Detecting silences in LV track...")
    lv_silences = detect_silences(
        LV_AUDIO_PATH, SILENCE_NOISE_THRESHOLD_DB,
        SILENCE_MINIMUM_DURATION_SECONDS, total_duration,
    )
    print(f"  Found {len(lv_silences)} silence regions")

    # Convert to speech regions
    ru_speech = silences_to_speech_regions(ru_silences, total_duration)
    lv_speech = silences_to_speech_regions(lv_silences, total_duration)

    print(f"\n  RU speech regions: {len(ru_speech)}")
    print(f"  LV speech regions: {len(lv_speech)}")

    # Show first 15 speech regions for each
    print("\n" + "=" * 80)
    print("RU SPEECH REGIONS (first 15)")
    print("-" * 60)
    for i, (start, end) in enumerate(ru_speech[:15]):
        duration = end - start
        print(f"  {i+1:3d}. {format_timestamp(start)} - {format_timestamp(end)}  ({duration:.1f}s)")

    print("\nLV SPEECH REGIONS (first 15)")
    print("-" * 60)
    for i, (start, end) in enumerate(lv_speech[:15]):
        duration = end - start
        print(f"  {i+1:3d}. {format_timestamp(start)} - {format_timestamp(end)}  ({duration:.1f}s)")

    # Overlap analysis
    print("\n" + "=" * 80)
    print("[3/4] OVERLAP ANALYSIS (both speaking simultaneously)")
    print("-" * 60)

    overlaps = find_overlaps(ru_speech, lv_speech)
    total_overlap = sum(d for _, _, d in overlaps)

    if overlaps:
        print(f"  Found {len(overlaps)} overlapping regions ({total_overlap:.1f}s total)")
        print()
        for i, (start, end, duration) in enumerate(overlaps[:20]):
            marker = " *** LONG" if duration > 2.0 else ""
            print(f"  {i+1:3d}. {format_timestamp(start)} - {format_timestamp(end)}  ({duration:.1f}s){marker}")
    else:
        print("  No overlaps found! Perfect complementary pattern.")

    # Gap analysis
    print("\n" + "=" * 80)
    print("[4/4] GAP ANALYSIS (neither speaking)")
    print("-" * 60)

    gaps = find_gaps(ru_speech, lv_speech, total_duration)
    total_gap = sum(d for _, _, d in gaps)

    if gaps:
        print(f"  Found {len(gaps)} gap regions ({total_gap:.1f}s total)")
        print()
        for i, (start, end, duration) in enumerate(gaps[:20]):
            marker = " *** LONG" if duration > 3.0 else ""
            print(f"  {i+1:3d}. {format_timestamp(start)} - {format_timestamp(end)}  ({duration:.1f}s){marker}")
    else:
        print("  No gaps found!")

    # Summary
    total_ru_speech = sum(e - s for s, e in ru_speech)
    total_lv_speech = sum(e - s for s, e in lv_speech)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 60)
    print(f"  RU total speech:     {total_ru_speech:6.1f}s ({total_ru_speech/total_duration*100:.1f}%)")
    print(f"  LV total speech:     {total_lv_speech:6.1f}s ({total_lv_speech/total_duration*100:.1f}%)")
    print(f"  Both speaking:       {total_overlap:6.1f}s ({total_overlap/total_duration*100:.1f}%)")
    print(f"  Neither speaking:    {total_gap:6.1f}s ({total_gap/total_duration*100:.1f}%)")
    print(f"  Analyzed:            {total_duration:6.1f}s")
    print()

    overlap_pct = total_overlap / total_duration * 100
    if overlap_pct < 5:
        print(f"  -> Overlap is {overlap_pct:.1f}% — good complementary pattern!")
    elif overlap_pct < 15:
        print(f"  -> Overlap is {overlap_pct:.1f}% — moderate, may need threshold tuning")
    else:
        print(f"  -> Overlap is {overlap_pct:.1f}% — high, threshold needs adjustment")

    print("=" * 80)


if __name__ == "__main__":
    main()
