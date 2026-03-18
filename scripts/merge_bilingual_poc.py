"""
Proof of concept: merge RU + LV subtitles into one bilingual SRT.

Uses FFmpeg silence detection + Transkriptor raw content.
Run from project root:
    python scripts/merge_bilingual_poc.py

Output: merged SRT next to audio files.
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.bilingual_subtitle_merger import (
    merge_bilingual_subtitles,
    load_transkriptor_content_from_json,
)

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(r"I:\2026-01-04 Pacelot mīlestības karogu\Audio RAW")

RU_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A04.wav"
LV_AUDIO = BASE_DIR / "2026-01-04 Pacelot mīlestības karogu_A03.wav"

RU_CONTENT_JSON = BASE_DIR / "transkriptor_raw_A04_RU.json"
LV_CONTENT_JSON = BASE_DIR / "transkriptor_raw_A03_LV.json"

OUTPUT_SRT = BASE_DIR.parent / "transcriptions" / "2026-01-04_Pacelot_merged_RU_LV.srt"

AUDIO_TO_VIDEO_OFFSET_SECONDS = 10.0  # same offset used in the pipeline


def main():
    print("Loading Transkriptor content...")
    ru_content = load_transkriptor_content_from_json(RU_CONTENT_JSON)
    lv_content = load_transkriptor_content_from_json(LV_CONTENT_JSON)
    print(f"  RU: {len(ru_content)} entries, LV: {len(lv_content)} entries")

    def on_progress(message: str) -> None:
        print(f"  {message}")

    print("\nMerging...")
    output_path = merge_bilingual_subtitles(
        primary_audio_path=RU_AUDIO,
        primary_language_tag="RU",
        primary_content=ru_content,
        secondary_audio_path=LV_AUDIO,
        secondary_language_tag="LV",
        secondary_content=lv_content,
        output_srt_path=OUTPUT_SRT,
        audio_to_video_offset_seconds=AUDIO_TO_VIDEO_OFFSET_SECONDS,
        noise_threshold_db=-25,
        on_progress=on_progress,
    )

    # Show first 20 entries from the merged SRT
    print(f"\n{'=' * 70}")
    print(f"PREVIEW — first 20 entries from {output_path.name}")
    print(f"{'=' * 70}")

    import srt
    subtitles = list(srt.parse(output_path.read_text(encoding="utf-8")))
    for subtitle in subtitles[:20]:
        start = str(subtitle.start)[:-3] if subtitle.start.microseconds else str(subtitle.start)
        end = str(subtitle.end)[:-3] if subtitle.end.microseconds else str(subtitle.end)
        print(f"\n{subtitle.index}")
        print(f"{start} --> {end}")
        print(subtitle.content)

    print(f"\n... ({len(subtitles)} total entries)")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
