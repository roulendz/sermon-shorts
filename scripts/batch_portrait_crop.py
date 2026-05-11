"""
Batch portrait crop + silence removal for all landscape clips in a folder.
Usage: python scripts/batch_portrait_crop.py
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.face_cropper import crop_segment_to_portrait, build_portrait_output_path
from pipeline.silence_remover import remove_silence_from_video, build_silence_removed_output_path

CLIPS_DIRECTORY = Path(r"I:\2026-05-03 Dievs meklē sēklu\clips\v4-2026-05-10")
MODEL_DIRECTORY = Path("D:/sermon-shorts/.models")
SPEED_MULTIPLIER = 1.3
SILENCE_MINIMUM_DURATION = 1.0

TAG_PATTERN = re.compile(r"\[(?:portrait|debug|trimmed|portrait-[\d.]+x)\]")


def is_landscape_source(file_path: Path) -> bool:
    return file_path.suffix == ".mp4" and not TAG_PATTERN.search(file_path.stem)


def get_duration(video_path: Path) -> float:
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main():
    all_files = [Path(f) for f in sorted(Path.iterdir(CLIPS_DIRECTORY)) if f.suffix == ".mp4"]
    source_clips = [f for f in all_files if is_landscape_source(f)]

    print(f"Found {len(source_clips)} landscape clips\n")

    total_start = time.time()
    results = []

    for index, clip_path in enumerate(source_clips, 1):
        clip_start = time.time()
        duration = get_duration(clip_path)
        print(f"[{index}/{len(source_clips)}] {clip_path.name} ({duration:.0f}s)")

        portrait_path = build_portrait_output_path(clip_path)

        try:
            print(f"  Portrait cropping (speed={SPEED_MULTIPLIER}x)...")
            crop_segment_to_portrait(
                source_video_path=clip_path,
                output_file_path=portrait_path,
                start_seconds=0.0,
                duration_seconds=duration,
                model_directory=MODEL_DIRECTORY,
                speed_multiplier=SPEED_MULTIPLIER,
                on_progress=lambda p: print(f"    {p:.0f}%", end="\r") if int(p) % 20 == 0 else None,
            )
            print(f"  Portrait done: {portrait_path.name}")

            print(f"  Removing silence (>{SILENCE_MINIMUM_DURATION}s)...")
            trimmed_path = build_silence_removed_output_path(portrait_path)
            remove_silence_from_video(
                source_video_path=portrait_path,
                output_file_path=trimmed_path,
                minimum_duration_seconds=SILENCE_MINIMUM_DURATION,
                on_progress=lambda msg: print(f"    {msg}"),
            )

            clip_elapsed = time.time() - clip_start
            results.append((clip_path.name, duration, clip_elapsed, True))
            print(f"  Done in {clip_elapsed:.0f}s\n")

        except Exception as error:
            clip_elapsed = time.time() - clip_start
            results.append((clip_path.name, duration, clip_elapsed, False))
            print(f"  FAILED: {error}\n")

    total_elapsed = time.time() - total_start

    print("=" * 60)
    print(f"BATCH COMPLETE: {total_elapsed:.0f}s total ({total_elapsed/60:.1f}min)")
    print(f"{'=' * 60}")

    success_count = sum(1 for r in results if r[3])
    total_source_duration = sum(r[1] for r in results)
    print(f"Success: {success_count}/{len(results)}")
    print(f"Total source duration: {total_source_duration:.0f}s ({total_source_duration/60:.1f}min)")
    print(f"Processing ratio: {total_elapsed/total_source_duration:.1f}x realtime\n")

    for name, duration, elapsed, success in results:
        status = "OK" if success else "FAIL"
        print(f"  [{status}] {elapsed:5.0f}s  {name}")


if __name__ == "__main__":
    main()
