"""
Generate TWO waveform images with human-readable timestamps in filenames:
1. RAW - all silence overlays as they were (green+pink overlapping)
2. PROCESSED - after /2 dynamic buffer (overlaps split at midpoint)

Both with 5-second timestamps on both tracks.
Same visual style as waveform_BEFORE_dynamic_buffer.png
"""

import subprocess
import struct
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from pipeline.silence_detection import detect_silence_regions, silence_regions_to_speech_regions
from pipeline.bilingual_subtitle_merger import (
    _remove_mic_bleed, _align_secondary_to_primary,
    _merge_pauses_where_other_track_is_silent,
    _fill_gaps_by_alternation,
)

BASE_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW")
RU_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A03.wav"
DIAG_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\diagnostics")

WINDOW_SECONDS = 600
TARGET_SAMPLE_RATE = 8000
NOISE_DB = -45
MIN_SILENCE = 1.0


def load_audio_as_mono(audio_path, duration_seconds):
    command = [
        "ffmpeg", "-y", "-t", str(duration_seconds),
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


def add_timestamps_5s(ax, window_seconds):
    for second in range(0, window_seconds + 1, 5):
        is_minute = second % 60 == 0
        is_30s = second % 30 == 0
        m = second // 60
        s = second % 60
        label = f"{m}:{s:02d}"
        if is_minute:
            ax.axvline(x=second, color="blue", linewidth=0.8, alpha=0.5)
            ax.text(second + 0.5, 0.92, label, fontsize=7, color="blue", alpha=0.7, fontweight="bold")
        elif is_30s:
            ax.axvline(x=second, color="blue", linewidth=0.4, alpha=0.35)
            ax.text(second + 0.5, 0.87, label, fontsize=6, color="blue", alpha=0.5)
        else:
            ax.axvline(x=second, color="gray", linewidth=0.2, alpha=0.2)
            ax.text(second + 0.3, 0.82, label, fontsize=5, color="gray", alpha=0.4)


def draw_waveform(ru_time, ru_norm, lv_time, lv_norm,
                  ru_silence_regions, lv_silence_regions,
                  ru_bleed, lv_bleed, lv_trims,
                  ru_false_silences, lv_false_silences,
                  title, output_path,
                  ru_peak=1.0, lv_peak=1.0):
    """
    Draw waveforms with silence overlays — same style as waveform_BEFORE_dynamic_buffer.png
    Green = RU silence (+ bleed + trims) on both tracks
    Pink = LV silence (+ false silences) on both tracks
    """
    fig, (ax_ru, ax_lv) = plt.subplots(2, 1, figsize=(50, 12), sharex=True)

    # RU waveform
    ax_ru.plot(ru_time, ru_norm, color="#d32f2f", linewidth=0.15, alpha=0.8)
    ax_ru.fill_between(ru_time, ru_norm, alpha=0.3, color="#d32f2f")
    ax_ru.set_ylabel("RU (Pastor)", fontsize=14, fontweight="bold", color="#d32f2f")
    ax_ru.set_ylim(-1, 1)
    ax_ru.grid(True, alpha=0.15)
    ax_ru.axhline(y=0, color="gray", linewidth=0.5)

    # LV waveform
    ax_lv.plot(lv_time, lv_norm, color="#1976d2", linewidth=0.15, alpha=0.8)
    ax_lv.fill_between(lv_time, lv_norm, alpha=0.3, color="#1976d2")
    ax_lv.set_ylabel("LV (Translator)", fontsize=14, fontweight="bold", color="#1976d2")
    ax_lv.set_ylim(-1, 1)
    ax_lv.set_xlabel("Time (seconds)", fontsize=12)
    ax_lv.grid(True, alpha=0.15)
    ax_lv.axhline(y=0, color="gray", linewidth=0.5)

    # Green = RU silence on both tracks
    for sil_s, sil_e in ru_silence_regions:
        w = sil_e - sil_s
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle((sil_s, -1), w, 2, linewidth=0, facecolor="#4caf50", alpha=0.25))

    # Pink = LV silence on both tracks
    for sil_s, sil_e in lv_silence_regions:
        w = sil_e - sil_s
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle((sil_s, -1), w, 2, linewidth=0, facecolor="#e91e63", alpha=0.15))

    # Bleed as green on both tracks
    for bl_s, bl_e in ru_bleed + lv_bleed:
        w = bl_e - bl_s
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle((bl_s, -1), w, 2, linewidth=0, facecolor="#4caf50", alpha=0.25))

    # LV alignment trims as green on both tracks
    for tr_s, tr_e in lv_trims:
        w = tr_e - tr_s
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle((tr_s, -1), w, 2, linewidth=0, facecolor="#4caf50", alpha=0.25))

    # False RU silences as pink on both (pastor paused)
    for fs_s, fs_e in ru_false_silences:
        w = fs_e - fs_s
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle((fs_s, -1), w, 2, linewidth=0, facecolor="#e91e63", alpha=0.15))

    # False LV silences as green on both (translator paused)
    for fs_s, fs_e in lv_false_silences:
        w = fs_e - fs_s
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle((fs_s, -1), w, 2, linewidth=0, facecolor="#4caf50", alpha=0.25))

    # Threshold lines — need original peak amplitudes (before normalization)
    # ru_norm = ru_samples / ru_peak, so ru_peak = max(abs(ru_samples))
    # threshold in normalized space = threshold_linear / ru_peak
    pass  # drawn per-image with correct peaks

    # Threshold lines
    threshold_linear = 10 ** (NOISE_DB / 20)
    ru_threshold_line = threshold_linear / ru_peak
    lv_threshold_line = threshold_linear / lv_peak
    ax_ru.axhline(y=ru_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_ru.axhline(y=-ru_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_ru.text(WINDOW_SECONDS - 25, ru_threshold_line + 0.02, f"{NOISE_DB}dB", fontsize=8, color="#ff9800", fontweight="bold")
    ax_lv.axhline(y=lv_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_lv.axhline(y=-lv_threshold_line, color="#ff9800", linewidth=1.0, linestyle="--", alpha=0.7)
    ax_lv.text(WINDOW_SECONDS - 25, lv_threshold_line + 0.02, f"{NOISE_DB}dB", fontsize=8, color="#ff9800", fontweight="bold")

    # 5-second timestamps on both tracks
    add_timestamps_5s(ax_ru, WINDOW_SECONDS)
    add_timestamps_5s(ax_lv, WINDOW_SECONDS)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading audio...")
    ru_samples = load_audio_as_mono(RU_AUDIO, WINDOW_SECONDS)
    lv_samples = load_audio_as_mono(LV_AUDIO, WINDOW_SECONDS)
    ru_time = np.arange(len(ru_samples)) / TARGET_SAMPLE_RATE
    lv_time = np.arange(len(lv_samples)) / TARGET_SAMPLE_RATE
    ru_peak = max(np.abs(ru_samples).max(), 0.01)
    lv_peak = max(np.abs(lv_samples).max(), 0.01)
    ru_norm = ru_samples / ru_peak
    lv_norm = lv_samples / lv_peak

    print("Detecting silences...")
    ru_silences_all = detect_silence_regions(str(RU_AUDIO), NOISE_DB, MIN_SILENCE)
    lv_silences_all = detect_silence_regions(str(LV_AUDIO), NOISE_DB, MIN_SILENCE)
    ru_silences = [(s, min(e, WINDOW_SECONDS)) for s, e in ru_silences_all if s < WINDOW_SECONDS]
    lv_silences = [(s, min(e, WINDOW_SECONDS)) for s, e in lv_silences_all if s < WINDOW_SECONDS]

    ru_speech_raw = [(s, e) for s, e in silence_regions_to_speech_regions(ru_silences_all, 4000) if e - s >= 0.5]
    lv_speech_raw = [(s, e) for s, e in silence_regions_to_speech_regions(lv_silences_all, 4000) if e - s >= 0.5]

    # Process
    ru_speech = ru_speech_raw  # RU priority
    lv_speech = _remove_mic_bleed(lv_speech_raw, ru_speech)
    lv_bleed = [r for r in lv_speech_raw if r not in lv_speech]

    lv_before_align = lv_speech[:]
    lv_speech = _align_secondary_to_primary(ru_speech, lv_speech)

    # Find trims
    lv_trims = []
    for old_s, old_e in lv_before_align:
        for new_s, new_e in lv_speech:
            if abs(old_e - new_e) < 0.5 and new_s > old_s + 0.1:
                lv_trims.append((old_s, new_s))
            if abs(old_s - new_s) < 0.5 and new_e < old_e - 0.1:
                lv_trims.append((new_e, old_e))

    # Classify false silences
    ru_false = []
    for s, e in ru_silences:
        has_lv = any(max(s, ls) < min(e, le) - 0.3 for ls, le in lv_speech)
        if not has_lv:
            ru_false.append((s, e))

    lv_false = []
    for s, e in lv_silences:
        has_ru = any(max(s, rs) < min(e, re) - 0.3 for rs, re in ru_speech)
        if not has_ru:
            lv_false.append((s, e))

    print(f"  RU silences: {len(ru_silences)}, LV silences: {len(lv_silences)}")
    print(f"  LV bleed: {len(lv_bleed)}, LV trims: {len(lv_trims)}")
    print(f"  RU false: {len(ru_false)}, LV false: {len(lv_false)}")

    # ── IMAGE 1: RAW (all overlays, overlaps visible) ──────────────────
    print("\n[1] RAW image...")
    draw_waveform(
        ru_time, ru_norm, lv_time, lv_norm,
        ru_silences, lv_silences,
        [], [], [],  # no bleed/trims in raw
        [], [],      # no false classification in raw
        "RAW Silence Detection — Green = RU silence | Pink = LV silence | Overlaps visible",
        DIAG_DIR / f"{timestamp}_1_RAW_silence_overlays.png",
        ru_peak=ru_peak, lv_peak=lv_peak,
    )

    # ── Fill gaps by alternation ──────────────────────────────────────
    ru_speech, lv_speech = _fill_gaps_by_alternation(ru_speech, lv_speech)
    print(f"  After gap filling: RU {len(ru_speech)}, LV {len(lv_speech)}")

    # For processed image: RU silence = LV speech regions, LV silence = RU speech regions
    ru_processed_silences = [(s, min(e, WINDOW_SECONDS)) for s, e in lv_speech if s < WINDOW_SECONDS]
    lv_processed_silences = [(s, min(e, WINDOW_SECONDS)) for s, e in ru_speech if s < WINDOW_SECONDS]

    # ── IMAGE 2: PROCESSED (no gaps, no overlaps) ──────────────────────
    print("\n[2] PROCESSED image (no gaps)...")
    draw_waveform(
        ru_time, ru_norm, lv_time, lv_norm,
        ru_processed_silences, lv_processed_silences,
        [], [], [],
        [], [],
        "PROCESSED — Green = LV speaking (RU silent) | Pink = RU speaking (LV silent) | No gaps",
        DIAG_DIR / f"{timestamp}_2_PROCESSED_no_gaps.png",
        ru_peak=ru_peak, lv_peak=lv_peak,
    )

    print(f"\nFiles at: {DIAG_DIR}")
    print(f"  {timestamp}_1_RAW_silence_overlays.png")
    print(f"  {timestamp}_2_PROCESSED_silence_overlays.png")


if __name__ == "__main__":
    main()
