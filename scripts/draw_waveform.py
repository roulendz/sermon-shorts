"""
Draw stacked waveforms: RU on top, LV on bottom.
First 5 minutes only.

Usage:
    python scripts/draw_waveform.py
"""

import subprocess
import struct
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW")
RU_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A03.wav"
OUTPUT_PNG = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\diagnostics\waveform_RU_LV.png")

WINDOW_SECONDS = 300  # first 5 min
TARGET_SAMPLE_RATE = 8000  # downsample for plotting


def load_audio_as_mono(audio_path, duration_seconds):
    """Use FFmpeg to extract raw PCM mono audio at target sample rate."""
    command = [
        "ffmpeg", "-y",
        "-t", str(duration_seconds),
        "-i", str(audio_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SAMPLE_RATE),
        "-ac", "1",
        "-f", "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr[-300:]}")

    raw_bytes = result.stdout
    sample_count = len(raw_bytes) // 2
    samples = np.array(struct.unpack(f"<{sample_count}h", raw_bytes[:sample_count * 2]), dtype=np.float32)
    # Normalize to -1..1
    samples = samples / 32768.0
    return samples


def main():
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    print("Loading RU audio...")
    ru_samples = load_audio_as_mono(RU_AUDIO, WINDOW_SECONDS)
    print(f"  {len(ru_samples)} samples ({len(ru_samples) / TARGET_SAMPLE_RATE:.0f}s)")

    print("Loading LV audio...")
    lv_samples = load_audio_as_mono(LV_AUDIO, WINDOW_SECONDS)
    print(f"  {len(lv_samples)} samples ({len(lv_samples) / TARGET_SAMPLE_RATE:.0f}s)")

    ru_time = np.arange(len(ru_samples)) / TARGET_SAMPLE_RATE
    lv_time = np.arange(len(lv_samples)) / TARGET_SAMPLE_RATE

    print("Drawing waveforms...")

    fig, (ax_ru, ax_lv) = plt.subplots(2, 1, figsize=(40, 10), sharex=True)

    # Normalize each track to fill the full height
    ru_max = max(np.abs(ru_samples).max(), 0.01)
    lv_max = max(np.abs(lv_samples).max(), 0.01)
    ru_normalized = ru_samples / ru_max
    lv_normalized = lv_samples / lv_max

    # RU waveform (top)
    ax_ru.plot(ru_time, ru_normalized, color="#d32f2f", linewidth=0.15, alpha=0.8)
    ax_ru.fill_between(ru_time, ru_normalized, alpha=0.3, color="#d32f2f")
    ax_ru.set_ylabel("RU (Pastor)", fontsize=14, fontweight="bold", color="#d32f2f")
    ax_ru.set_ylim(-1, 1)
    ax_ru.grid(True, alpha=0.3)
    ax_ru.axhline(y=0, color="gray", linewidth=0.5)

    # Add minute markers
    for minute in range(0, WINDOW_SECONDS // 60 + 1):
        second = minute * 60
        ax_ru.axvline(x=second, color="blue", linewidth=0.5, alpha=0.4)
        ax_ru.text(second + 1, 0.9, f"{minute}:00", fontsize=8, color="blue", alpha=0.6)

    # LV waveform (bottom)
    ax_lv.plot(lv_time, lv_normalized, color="#1976d2", linewidth=0.15, alpha=0.8)
    ax_lv.fill_between(lv_time, lv_normalized, alpha=0.3, color="#1976d2")
    ax_lv.set_ylabel("LV (Translator)", fontsize=14, fontweight="bold", color="#1976d2")
    ax_lv.set_ylim(-1, 1)
    ax_lv.set_xlabel("Time (seconds)", fontsize=12)
    ax_lv.grid(True, alpha=0.3)
    ax_lv.axhline(y=0, color="gray", linewidth=0.5)

    for minute in range(0, WINDOW_SECONDS // 60 + 1):
        second = minute * 60
        ax_lv.axvline(x=second, color="blue", linewidth=0.5, alpha=0.4)

    fig.suptitle("Sermon Audio Waveforms — RU (Pastor) vs LV (Translator)", fontsize=16, fontweight="bold")
    fig.tight_layout()

    fig.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUTPUT_PNG}")
    print(f"Size: {OUTPUT_PNG.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
