"""
Generate TWO waveform images:
1. RAW - original silence detection overlays (green=RU silence, pink=LV silence)
2. PROCESSED - after /2 dynamic buffer split

Both with 5-second timestamps on both tracks.
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
from pipeline.bilingual_subtitle_merger import (
    _remove_mic_bleed, _align_secondary_to_primary, _split_overlaps_at_midpoint,
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


def add_timestamps(ax, window_seconds):
    """Add 5-second markers with timestamps on both tracks."""
    for second in range(0, window_seconds + 1, 5):
        is_minute = second % 60 == 0
        is_30s = second % 30 == 0
        if is_minute:
            ax.axvline(x=second, color="blue", linewidth=0.8, alpha=0.5)
            ax.text(second + 0.5, 0.9, f"{second // 60}:{second % 60:02d}",
                    fontsize=7, color="blue", alpha=0.7, fontweight="bold")
        elif is_30s:
            ax.axvline(x=second, color="blue", linewidth=0.4, alpha=0.35)
            ax.text(second + 0.5, 0.85, f"{second // 60}:{second % 60:02d}",
                    fontsize=6, color="blue", alpha=0.5)
        else:
            ax.axvline(x=second, color="gray", linewidth=0.2, alpha=0.2)
            ax.text(second + 0.3, 0.8, f"{second // 60}:{second % 60:02d}",
                    fontsize=5, color="gray", alpha=0.4)


def draw_image_with_silences(ru_time, ru_normalized, lv_time, lv_normalized,
                             ru_silences, lv_silences, title, output_path):
    """Draw waveform with SILENCE region overlays (green=RU silence, pink=LV silence) on both tracks."""
    fig, (ax_ru, ax_lv) = plt.subplots(2, 1, figsize=(50, 12), sharex=True)

    # RU waveform
    ax_ru.plot(ru_time, ru_normalized, color="#d32f2f", linewidth=0.15, alpha=0.8)
    ax_ru.fill_between(ru_time, ru_normalized, alpha=0.3, color="#d32f2f")
    ax_ru.set_ylabel("RU (Pastor)", fontsize=14, fontweight="bold", color="#d32f2f")
    ax_ru.set_ylim(-1, 1)
    ax_ru.grid(True, alpha=0.15)
    ax_ru.axhline(y=0, color="gray", linewidth=0.5)

    # LV waveform
    ax_lv.plot(lv_time, lv_normalized, color="#1976d2", linewidth=0.15, alpha=0.8)
    ax_lv.fill_between(lv_time, lv_normalized, alpha=0.3, color="#1976d2")
    ax_lv.set_ylabel("LV (Translator)", fontsize=14, fontweight="bold", color="#1976d2")
    ax_lv.set_ylim(-1, 1)
    ax_lv.set_xlabel("Time (seconds)", fontsize=12)
    ax_lv.grid(True, alpha=0.15)
    ax_lv.axhline(y=0, color="gray", linewidth=0.5)

    # Green = RU silence (where pastor is NOT speaking) on both tracks
    for sil_start, sil_end in ru_silences:
        if sil_start >= WINDOW_SECONDS:
            continue
        width = min(sil_end, WINDOW_SECONDS) - sil_start
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle(
                (sil_start, -1), width, 2,
                linewidth=0, facecolor="#4caf50", alpha=0.25,
            ))

    # Pink = LV silence (where translator is NOT speaking) on both tracks
    for sil_start, sil_end in lv_silences:
        if sil_start >= WINDOW_SECONDS:
            continue
        width = min(sil_end, WINDOW_SECONDS) - sil_start
        for ax in [ax_ru, ax_lv]:
            ax.add_patch(patches.Rectangle(
                (sil_start, -1), width, 2,
                linewidth=0, facecolor="#e91e63", alpha=0.15,
            ))

    # Timestamps on both tracks
    add_timestamps(ax_ru, WINDOW_SECONDS)
    add_timestamps(ax_lv, WINDOW_SECONDS)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def main():
    print("Loading audio...")
    ru_samples = load_audio_as_mono(RU_AUDIO, WINDOW_SECONDS)
    lv_samples = load_audio_as_mono(LV_AUDIO, WINDOW_SECONDS)

    ru_time = np.arange(len(ru_samples)) / TARGET_SAMPLE_RATE
    lv_time = np.arange(len(lv_samples)) / TARGET_SAMPLE_RATE

    ru_max = max(np.abs(ru_samples).max(), 0.01)
    lv_max = max(np.abs(lv_samples).max(), 0.01)
    ru_normalized = ru_samples / ru_max
    lv_normalized = lv_samples / lv_max

    print("Detecting silences...")
    ru_silences = detect_silence_regions(str(RU_AUDIO), NOISE_DB, MIN_SILENCE)
    lv_silences = detect_silence_regions(str(LV_AUDIO), NOISE_DB, MIN_SILENCE)

    ru_speech_raw = [(s, e) for s, e in silence_regions_to_speech_regions(ru_silences, 4000) if e - s >= 0.5]
    lv_speech_raw = [(s, e) for s, e in silence_regions_to_speech_regions(lv_silences, 4000) if e - s >= 0.5]

    print(f"  RAW: RU {len(ru_speech_raw)} speech, LV {len(lv_speech_raw)} speech")

    # Raw silence regions (clipped to window)
    ru_silences_window = [(s, min(e, WINDOW_SECONDS)) for s, e in ru_silences if s < WINDOW_SECONDS]
    lv_silences_window = [(s, min(e, WINDOW_SECONDS)) for s, e in lv_silences if s < WINDOW_SECONDS]

    # ── IMAGE 1: RAW silence overlays ───────────────────────────────────
    print("\n[1] RAW silence overlays (green=RU silence, pink=LV silence)...")
    draw_image_with_silences(
        ru_time, ru_normalized, lv_time, lv_normalized,
        ru_silences_window, lv_silences_window,
        "RAW — Green = RU silence | Pink = LV silence | Overlaps visible",
        DIAG_DIR / "waveform_1_RAW.png",
    )

    # ── Process: bleed removal + alignment + buffer ─────────────────────
    print("\n[2] Processing speech regions...")
    ru_speech = ru_speech_raw  # RU priority — never remove
    lv_speech = _remove_mic_bleed(lv_speech_raw, ru_speech)
    print(f"  After LV bleed removal: {len(lv_speech)} LV regions")

    lv_speech = _align_secondary_to_primary(ru_speech, lv_speech)
    print(f"  After LV alignment: {len(lv_speech)} LV regions")

    ru_speech, lv_speech = _split_overlaps_at_midpoint(ru_speech, lv_speech)
    print(f"  After dynamic buffer: RU {len(ru_speech)}, LV {len(lv_speech)}")

    # Convert processed speech back to silence regions for visualization
    # RU silence = gaps between RU speech regions
    ru_processed_silences = []
    ru_sorted = sorted(ru_speech)
    for i in range(len(ru_sorted) - 1):
        gap_start = ru_sorted[i][1]
        gap_end = ru_sorted[i + 1][0]
        if gap_end > gap_start + 0.05:
            ru_processed_silences.append((gap_start, min(gap_end, WINDOW_SECONDS)))
    # Add silence at start and end
    if ru_sorted and ru_sorted[0][0] > 0.1:
        ru_processed_silences.insert(0, (0, ru_sorted[0][0]))
    if ru_sorted and ru_sorted[-1][1] < WINDOW_SECONDS - 0.1:
        ru_processed_silences.append((ru_sorted[-1][1], WINDOW_SECONDS))

    lv_processed_silences = []
    lv_sorted = sorted(lv_speech)
    for i in range(len(lv_sorted) - 1):
        gap_start = lv_sorted[i][1]
        gap_end = lv_sorted[i + 1][0]
        if gap_end > gap_start + 0.05:
            lv_processed_silences.append((gap_start, min(gap_end, WINDOW_SECONDS)))
    if lv_sorted and lv_sorted[0][0] > 0.1:
        lv_processed_silences.insert(0, (0, lv_sorted[0][0]))
    if lv_sorted and lv_sorted[-1][1] < WINDOW_SECONDS - 0.1:
        lv_processed_silences.append((lv_sorted[-1][1], WINDOW_SECONDS))

    print(f"  Processed silences: RU {len(ru_processed_silences)}, LV {len(lv_processed_silences)}")

    # ── IMAGE 2: PROCESSED silence overlays ─────────────────────────────
    print("\n[3] PROCESSED silence overlays (after /2 buffer)...")
    draw_image_with_silences(
        ru_time, ru_normalized, lv_time, lv_normalized,
        ru_processed_silences, lv_processed_silences,
        "PROCESSED — Green = RU silence | Pink = LV silence | After bleed removal + alignment + /2 buffer",
        DIAG_DIR / "waveform_2_PROCESSED.png",
    )

    print(f"\nBoth images at: {DIAG_DIR}")


if __name__ == "__main__":
    main()
