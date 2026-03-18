"""
Diagnostic: output silence/speech regions for RU and LV, compare merge quality,
and validate SRT timestamps against actual audio.

Generates:
  - {base}/diagnostics/ru_silence_regions.txt
  - {base}/diagnostics/ru_speech_regions.txt
  - {base}/diagnostics/lv_silence_regions.txt
  - {base}/diagnostics/lv_speech_regions.txt
  - {base}/diagnostics/merged_timeline.txt (interleaved RU/LV/SILENCE)
  - {base}/diagnostics/srt_vs_audio_validation.txt

Usage:
    python scripts/speech_silence_diagnostic.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

import srt
from pipeline.silence_detection import (
    detect_silence_regions,
    silence_regions_to_speech_regions,
    get_audio_duration_seconds,
)

BASE_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW")
DIAG_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\diagnostics")

RU_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A03.wav"
SRT_FILE = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\transcriptions\2026-01-04_Pacelot_merged_RU_LV_v2.srt")

NOISE_DB = -25
MIN_SILENCE = 1.0
WINDOW = 300  # first 5 min
OFFSET = 10.0  # video offset


def fmt(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:06.3f}"


def write_regions_file(filepath, regions, label):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {label}\n")
        f.write(f"# Format: INDEX | START | END | DURATION\n")
        f.write(f"# Times are AUDIO time (add {OFFSET}s for video time)\n")
        f.write(f"# Total regions: {len(regions)}\n")
        f.write(f"# Total duration: {sum(e - s for s, e in regions):.1f}s\n\n")
        for i, (start, end) in enumerate(regions, 1):
            if start >= WINDOW:
                break
            dur = end - start
            f.write(f"{i:4d} | {fmt(start)} | {fmt(end)} | {dur:6.2f}s\n")
    print(f"  Written: {filepath.name}")


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SPEECH/SILENCE DIAGNOSTIC")
    print(f"Threshold: {NOISE_DB}dB, Min silence: {MIN_SILENCE}s")
    print(f"Window: first {WINDOW}s, Video offset: {OFFSET}s")
    print("=" * 80)

    # ── Detect silence and speech ───────────────────────────────────────
    print("\n[1] Detecting silence/speech regions...")

    ru_duration = get_audio_duration_seconds(str(RU_AUDIO))
    ru_silences_all = detect_silence_regions(str(RU_AUDIO), NOISE_DB, MIN_SILENCE)
    ru_silences = [(s, e) for s, e in ru_silences_all if s < WINDOW]
    ru_speech_all = silence_regions_to_speech_regions(ru_silences_all, ru_duration)
    ru_speech = [(s, e) for s, e in ru_speech_all if s < WINDOW]

    lv_duration = get_audio_duration_seconds(str(LV_AUDIO))
    lv_silences_all = detect_silence_regions(str(LV_AUDIO), NOISE_DB, MIN_SILENCE)
    lv_silences = [(s, e) for s, e in lv_silences_all if s < WINDOW]
    lv_speech_all = silence_regions_to_speech_regions(lv_silences_all, lv_duration)
    lv_speech = [(s, e) for s, e in lv_speech_all if s < WINDOW]

    print(f"  RU: {len(ru_silences)} silences, {len(ru_speech)} speech regions")
    print(f"  LV: {len(lv_silences)} silences, {len(lv_speech)} speech regions")

    # ── Write individual files ──────────────────────────────────────────
    print("\n[2] Writing region files...")
    write_regions_file(DIAG_DIR / "ru_silence_regions.txt", ru_silences, "RU SILENCE REGIONS")
    write_regions_file(DIAG_DIR / "ru_speech_regions.txt", ru_speech, "RU SPEECH REGIONS")
    write_regions_file(DIAG_DIR / "lv_silence_regions.txt", lv_silences, "LV SILENCE REGIONS")
    write_regions_file(DIAG_DIR / "lv_speech_regions.txt", lv_speech, "LV SPEECH REGIONS")

    # ── Merged timeline ─────────────────────────────────────────────────
    print("\n[3] Building merged timeline...")

    # Create events at 0.1s resolution
    timeline = []
    for s, e in ru_speech:
        if s < WINDOW:
            timeline.append((s, min(e, WINDOW), "RU_SPEECH"))
    for s, e in lv_speech:
        if s < WINDOW:
            timeline.append((s, min(e, WINDOW), "LV_SPEECH"))
    timeline.sort(key=lambda x: x[0])

    with open(DIAG_DIR / "merged_timeline.txt", "w", encoding="utf-8") as f:
        f.write(f"# MERGED TIMELINE - RU and LV speech regions interleaved\n")
        f.write(f"# Shows who is speaking when, and gaps where nobody speaks\n")
        f.write(f"# Audio time (add {OFFSET}s for video time)\n\n")

        previous_end = 0.0
        for start, end, label in timeline:
            # Show gap before this region
            if start > previous_end + 0.05:
                gap_dur = start - previous_end
                f.write(f"  {'---':>10}  | {fmt(previous_end)} - {fmt(start)} | {gap_dur:5.2f}s | SILENCE (both)\n")

            dur = end - start
            marker = ">>>" if dur < 0.5 else "   "
            f.write(f"{marker} {label:>10}  | {fmt(start)} - {fmt(end)} | {dur:5.2f}s |\n")
            previous_end = max(previous_end, end)

    print(f"  Written: merged_timeline.txt")

    # ── Compare RU/LV merge quality ─────────────────────────────────────
    print("\n[4] Comparing RU/LV merge quality...")

    overlaps = []
    for ru_s, ru_e in ru_speech:
        for lv_s, lv_e in lv_speech:
            o_start = max(ru_s, lv_s)
            o_end = min(ru_e, lv_e)
            if o_start < o_end:
                overlaps.append((o_start, o_end, o_end - o_start))

    total_ru = sum(e - s for s, e in ru_speech if s < WINDOW)
    total_lv = sum(e - s for s, e in lv_speech if s < WINDOW)
    total_overlap = sum(d for _, _, d in overlaps)

    # Find gaps where neither speaks
    all_speech = sorted(ru_speech + lv_speech)
    merged_speech = []
    for s, e in all_speech:
        if s >= WINDOW:
            break
        if merged_speech and s <= merged_speech[-1][1]:
            merged_speech[-1] = (merged_speech[-1][0], max(merged_speech[-1][1], e))
        else:
            merged_speech.append((s, min(e, WINDOW)))

    gaps = []
    prev = 0.0
    for s, e in merged_speech:
        if s > prev + 0.1:
            gaps.append((prev, s, s - prev))
        prev = e

    with open(DIAG_DIR / "merge_quality.txt", "w", encoding="utf-8") as f:
        f.write("# RU/LV MERGE QUALITY ANALYSIS\n\n")
        f.write(f"RU total speech: {total_ru:.1f}s ({total_ru/WINDOW*100:.1f}%)\n")
        f.write(f"LV total speech: {total_lv:.1f}s ({total_lv/WINDOW*100:.1f}%)\n")
        f.write(f"Both speaking:   {total_overlap:.1f}s ({total_overlap/WINDOW*100:.1f}%)\n")
        f.write(f"Neither speaking:{sum(d for _,_,d in gaps):.1f}s\n\n")

        f.write(f"OVERLAPS (both speaking at once): {len(overlaps)}\n")
        for i, (s, e, d) in enumerate(overlaps, 1):
            f.write(f"  {i:3d}. {fmt(s)} - {fmt(e)} ({d:.2f}s)\n")

        f.write(f"\nGAPS (neither speaking): {len(gaps)}\n")
        for i, (s, e, d) in enumerate(gaps, 1):
            f.write(f"  {i:3d}. {fmt(s)} - {fmt(e)} ({d:.2f}s)\n")

    print(f"  Written: merge_quality.txt")
    print(f"  Overlaps: {len(overlaps)} ({total_overlap:.1f}s)")
    print(f"  Gaps: {len(gaps)} ({sum(d for _,_,d in gaps):.1f}s)")

    # ── Validate SRT against audio ──────────────────────────────────────
    print("\n[5] Validating SRT timestamps against audio...")

    srt_text = SRT_FILE.read_text(encoding="utf-8")
    subtitles = list(srt.parse(srt_text))

    with open(DIAG_DIR / "srt_vs_audio_validation.txt", "w", encoding="utf-8") as f:
        f.write("# SRT vs AUDIO VALIDATION\n")
        f.write("# Checks if each subtitle's time range falls within its language's speech regions\n")
        f.write("# and does NOT fall within the other language's speech regions\n\n")

        issues = 0
        for sub in subtitles:
            video_start = sub.start.total_seconds()
            video_end = sub.end.total_seconds()
            if video_start > WINDOW + OFFSET:
                break

            audio_start = video_start - OFFSET
            audio_end = video_end - OFFSET
            if audio_start < 0:
                continue

            is_ru = "[RU]" in sub.content
            is_lv = "[LV]" in sub.content
            lang = "RU" if is_ru else "LV"

            own_speech = ru_speech if is_ru else lv_speech
            other_speech = lv_speech if is_ru else ru_speech
            other_lang = "LV" if is_ru else "RU"

            # How much falls in own speech?
            own_covered = 0.0
            for s, e in own_speech:
                o_s = max(audio_start, s)
                o_e = min(audio_end, e)
                if o_s < o_e:
                    own_covered += o_e - o_s

            # How much falls in other's speech?
            other_covered = 0.0
            for s, e in other_speech:
                o_s = max(audio_start, s)
                o_e = min(audio_end, e)
                if o_s < o_e:
                    other_covered += o_e - o_s

            sub_dur = audio_end - audio_start
            silence_dur = sub_dur - own_covered - other_covered
            text_preview = sub.content[:60]

            # Report issues
            status = "OK"
            if other_covered > 0.3:
                status = f"!! {other_covered:.1f}s in {other_lang} speech"
                issues += 1
            elif own_covered < sub_dur * 0.3 and sub_dur > 1.0:
                status = f"?? only {own_covered:.1f}s of {sub_dur:.1f}s in {lang} speech"
                issues += 1

            f.write(f"#{sub.index:4d} [{lang}] video {fmt(video_start)}-{fmt(video_end)} "
                    f"audio {fmt(audio_start)}-{fmt(audio_end)} | "
                    f"own:{own_covered:.1f}s other:{other_covered:.1f}s silence:{silence_dur:.1f}s "
                    f"| {status}\n")

            # Show which speech regions overlap
            if status != "OK":
                f.write(f"       Text: {text_preview}\n")
                f.write(f"       {lang} speech in range:\n")
                for s, e in own_speech:
                    if s < audio_end and e > audio_start:
                        f.write(f"         {fmt(s)} - {fmt(e)} ({e-s:.2f}s)\n")
                f.write(f"       {other_lang} speech in range:\n")
                for s, e in other_speech:
                    if s < audio_end and e > audio_start:
                        f.write(f"         {fmt(s)} - {fmt(e)} ({e-s:.2f}s)\n")
                f.write("\n")

        f.write(f"\nTotal issues: {issues}\n")

    print(f"  Written: srt_vs_audio_validation.txt")
    print(f"  Issues: {issues}")

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"All files in: {DIAG_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
