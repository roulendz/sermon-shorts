"""
Draw stacked waveforms with silence detection overlay.
RU on top, LV on bottom. Green translucent rectangles for detected silence.

Usage:
    python scripts/draw_waveform_with_silence.py
"""

import subprocess
import struct
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from pipeline.silence_detection import detect_silence_regions, silence_regions_to_speech_regions

BASE_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW")
RU_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A03.wav"
OUTPUT_PNG = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\diagnostics\waveform_RU_LV_silence.png")

WINDOW_SECONDS = 600
TARGET_SAMPLE_RATE = 8000
NOISE_DB = -45
MIN_SILENCE = 1.0


def load_audio_as_mono(audio_path, duration_seconds):
    command = [
        "ffmpeg", "-y",
        "-t", str(duration_seconds),
        "-i", str(audio_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1", "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True)
    raw_bytes = result.stdout
    sample_count = len(raw_bytes) // 2
    samples = np.array(struct.unpack(f"<{sample_count}h", raw_bytes[:sample_count * 2]), dtype=np.float32)
    return samples / 32768.0


def main():
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    print("Loading audio...")
    ru_samples = load_audio_as_mono(RU_AUDIO, WINDOW_SECONDS)
    lv_samples = load_audio_as_mono(LV_AUDIO, WINDOW_SECONDS)

    ru_time = np.arange(len(ru_samples)) / TARGET_SAMPLE_RATE
    lv_time = np.arange(len(lv_samples)) / TARGET_SAMPLE_RATE

    # Normalize
    ru_max = max(np.abs(ru_samples).max(), 0.01)
    lv_max = max(np.abs(lv_samples).max(), 0.01)
    ru_normalized = ru_samples / ru_max
    lv_normalized = lv_samples / lv_max

    # Calculate threshold lines in normalized space
    # -25dB means 10^(-25/20) = ~0.0562 of full scale
    threshold_linear = 10 ** (NOISE_DB / 20)  # relative to full scale (1.0)
    ru_threshold_normalized = threshold_linear / ru_max * 32768.0 / 32768.0
    lv_threshold_normalized = threshold_linear / lv_max * 32768.0 / 32768.0
    # Actually: threshold is relative to digital full scale (32768)
    # In normalized space (divided by track max), the line is at threshold_linear / (max/32768)
    ru_threshold_line = threshold_linear * 32768.0 / (ru_max * 32768.0)
    lv_threshold_line = threshold_linear * 32768.0 / (lv_max * 32768.0)
    # Simplify: threshold_linear / ru_max * 1.0... no.
    # ru_samples are already /32768, then /ru_max. So in normalized space:
    # silence = amplitude < threshold_linear (in 0..1 space before normalization)
    # In normalized space: threshold_linear / ru_max
    ru_threshold_line = threshold_linear / ru_max
    lv_threshold_line = threshold_linear / lv_max

    print("Detecting RU silences...")
    ru_silences = detect_silence_regions(str(RU_AUDIO), NOISE_DB, MIN_SILENCE)
    ru_silences_window = [(s, min(e, WINDOW_SECONDS)) for s, e in ru_silences if s < WINDOW_SECONDS]
    print(f"  {len(ru_silences_window)} RU silence regions (raw)")

    print("Detecting LV silences...")
    lv_silences = detect_silence_regions(str(LV_AUDIO), NOISE_DB, MIN_SILENCE)
    lv_silences_window = [(s, min(e, WINDOW_SECONDS)) for s, e in lv_silences if s < WINDOW_SECONDS]
    print(f"  {len(lv_silences_window)} LV silence regions (raw)")

    # Get speech regions (inverse of silence)
    ru_speech_raw = silence_regions_to_speech_regions(ru_silences, WINDOW_SECONDS)
    lv_speech_raw = silence_regions_to_speech_regions(lv_silences, WINDOW_SECONDS)

    # Remove mic bleed: speech that's mostly during other track's speech
    from pipeline.bilingual_subtitle_merger import _remove_mic_bleed, _align_secondary_to_primary
    ru_speech = _remove_mic_bleed(ru_speech_raw, lv_speech_raw)
    lv_speech = _remove_mic_bleed(lv_speech_raw, ru_speech_raw)

    # Find bleed regions (removed speech)
    ru_bleed = [r for r in ru_speech_raw if r not in ru_speech]
    lv_bleed = [r for r in lv_speech_raw if r not in lv_speech]
    print(f"  RU mic bleed removed: {len(ru_bleed)} regions")
    print(f"  LV mic bleed removed: {len(lv_bleed)} regions")

    # RU priority: trim LV on both sides where RU overlaps
    lv_speech_before_align = lv_speech[:]
    lv_speech = _align_secondary_to_primary(ru_speech, lv_speech)

    # Find alignment trims (trimmed parts from start and end become silence)
    lv_alignment_trims = []
    for old_s, old_e in lv_speech_before_align:
        for new_s, new_e in lv_speech:
            # Start trimmed (pushed forward)
            if abs(old_e - new_e) < 0.5 and new_s > old_s + 0.1:
                lv_alignment_trims.append((old_s, new_s))
            # End trimmed (pulled backward)
            if abs(old_s - new_s) < 0.5 and new_e < old_e - 0.1:
                lv_alignment_trims.append((new_e, old_e))
    # Also regions that were completely removed
    lv_aligned_set = set(lv_speech)
    for old_region in lv_speech_before_align:
        if old_region not in lv_aligned_set:
            found_partial = any(
                abs(old_region[0] - new[0]) < 0.5 or abs(old_region[1] - new[1]) < 0.5
                for new in lv_speech
            )
            if not found_partial:
                lv_alignment_trims.append(old_region)
    print(f"  LV alignment trims (RU priority): {len(lv_alignment_trims)} regions")

    # Rebuild silences from cleaned speech to classify real vs false
    # Filter: only keep RU silence where LV actually has speech (real speaker switch)
    # Remove RU silence where LV is ALSO silent (false detection — just a pause)
    ru_real_silences = []
    ru_false_silences = []
    for sil_start, sil_end in ru_silences_window:
        lv_has_speech = False
        for sp_start, sp_end in lv_speech:
            overlap_start = max(sil_start, sp_start)
            overlap_end = min(sil_end, sp_end)
            if overlap_end - overlap_start > 0.3:
                lv_has_speech = True
                break
        if lv_has_speech:
            ru_real_silences.append((sil_start, sil_end))
        else:
            ru_false_silences.append((sil_start, sil_end))

    # Same for LV
    lv_real_silences = []
    lv_false_silences = []
    for sil_start, sil_end in lv_silences_window:
        ru_has_speech = False
        for sp_start, sp_end in ru_speech:
            overlap_start = max(sil_start, sp_start)
            overlap_end = min(sil_end, sp_end)
            if overlap_end - overlap_start > 0.3:
                ru_has_speech = True
                break
        if ru_has_speech:
            lv_real_silences.append((sil_start, sil_end))
        else:
            lv_false_silences.append((sil_start, sil_end))

    print(f"  RU silence: {len(ru_real_silences)} real (LV speaking) + {len(ru_false_silences)} false (both silent)")
    print(f"  LV silence: {len(lv_real_silences)} real (RU speaking) + {len(lv_false_silences)} false (both silent)")

    print("Drawing...")
    fig, (ax_ru, ax_lv) = plt.subplots(2, 1, figsize=(40, 10), sharex=True)

    # RU waveform
    ax_ru.plot(ru_time, ru_normalized, color="#d32f2f", linewidth=0.15, alpha=0.8)
    ax_ru.fill_between(ru_time, ru_normalized, alpha=0.3, color="#d32f2f")
    ax_ru.set_ylabel("RU (Pastor)", fontsize=14, fontweight="bold", color="#d32f2f")
    ax_ru.set_ylim(-1, 1)
    ax_ru.grid(True, alpha=0.3)
    ax_ru.axhline(y=0, color="gray", linewidth=0.5)

    # Overlay REAL RU silence as green (LV is speaking during these)
    for silence_start, silence_end in ru_real_silences:
        width = silence_end - silence_start
        rect = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_ru.add_patch(rect)

    # Threshold lines
    ax_ru.axhline(y=ru_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_ru.axhline(y=-ru_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_ru.text(WINDOW_SECONDS - 15, ru_threshold_line + 0.03, f"{NOISE_DB}dB", fontsize=8, color="#ff9800", fontweight="bold")

    # Minute markers
    for minute in range(0, WINDOW_SECONDS // 60 + 1):
        second = minute * 60
        ax_ru.axvline(x=second, color="blue", linewidth=0.5, alpha=0.4)
        ax_ru.text(second + 1, 0.9, f"{minute}:00", fontsize=8, color="blue", alpha=0.6)

    # LV waveform
    ax_lv.plot(lv_time, lv_normalized, color="#1976d2", linewidth=0.15, alpha=0.8)
    ax_lv.fill_between(lv_time, lv_normalized, alpha=0.3, color="#1976d2")
    ax_lv.set_ylabel("LV (Translator)", fontsize=14, fontweight="bold", color="#1976d2")
    ax_lv.set_ylim(-1, 1)
    ax_lv.set_xlabel("Time (seconds)", fontsize=12)
    ax_lv.grid(True, alpha=0.3)
    ax_lv.axhline(y=0, color="gray", linewidth=0.5)

    # RU mic bleed as green on both tracks (treated as silence)
    for bleed_start, bleed_end in ru_bleed:
        width = bleed_end - bleed_start
        rect_ru = patches.Rectangle(
            (bleed_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_ru.add_patch(rect_ru)
        rect_lv = patches.Rectangle(
            (bleed_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_lv.add_patch(rect_lv)

    # FALSE RU silences as yellow (both tracks silent — just a pause, should be merged)
    for silence_start, silence_end in ru_false_silences:
        width = silence_end - silence_start
        rect = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#ffeb3b", alpha=0.35,
        )
        ax_ru.add_patch(rect)

    # Threshold lines on LV
    ax_lv.axhline(y=lv_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_lv.axhline(y=-lv_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_lv.text(WINDOW_SECONDS - 15, lv_threshold_line + 0.03, f"{NOISE_DB}dB", fontsize=8, color="#ff9800", fontweight="bold")

    # Overlay REAL RU silence (green) on LV track
    for silence_start, silence_end in ru_real_silences:
        width = silence_end - silence_start
        rect = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_lv.add_patch(rect)

    # FALSE RU silences as yellow on LV track too
    for silence_start, silence_end in ru_false_silences:
        width = silence_end - silence_start
        rect = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#ffeb3b", alpha=0.35,
        )
        ax_lv.add_patch(rect)

    # Overlay REAL LV silence (pink) on BOTH tracks
    for silence_start, silence_end in lv_real_silences:
        width = silence_end - silence_start
        rect_ru = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#e91e63", alpha=0.15,
        )
        ax_ru.add_patch(rect_ru)
        rect_lv = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#e91e63", alpha=0.15,
        )
        ax_lv.add_patch(rect_lv)

    for minute in range(0, WINDOW_SECONDS // 60 + 1):
        second = minute * 60
        ax_lv.axvline(x=second, color="blue", linewidth=0.5, alpha=0.4)

    # LV alignment trims as green on both tracks (RU talked over LV start)
    for trim_start, trim_end in lv_alignment_trims:
        width = trim_end - trim_start
        rect_ru = patches.Rectangle(
            (trim_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_ru.add_patch(rect_ru)
        rect_lv = patches.Rectangle(
            (trim_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_lv.add_patch(rect_lv)

    # LV mic bleed as green on both tracks (treated as silence)
    for bleed_start, bleed_end in lv_bleed:
        width = bleed_end - bleed_start
        rect_ru = patches.Rectangle(
            (bleed_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_ru.add_patch(rect_ru)
        rect_lv = patches.Rectangle(
            (bleed_start, -1), width, 2,
            linewidth=0, facecolor="#4caf50", alpha=0.25,
        )
        ax_lv.add_patch(rect_lv)

    # FALSE LV silences as yellow on both tracks
    for silence_start, silence_end in lv_false_silences:
        width = silence_end - silence_start
        rect_ru = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#ffeb3b", alpha=0.35,
        )
        ax_ru.add_patch(rect_ru)
        rect_lv = patches.Rectangle(
            (silence_start, -1), width, 2,
            linewidth=0, facecolor="#ffeb3b", alpha=0.35,
        )
        ax_lv.add_patch(rect_lv)

    fig.suptitle(
        "Green=real silence | Pink=real LV silence | Yellow=false silence (merge) | Orange=mic bleed (removed)",
        fontsize=16, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")

    print(f"\nSaved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
