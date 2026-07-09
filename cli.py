"""
cli.py -- Headless driver for the Sermon Shorts viral-clip pipeline.

Reproduces, non-interactively, the exact stage order the Textual TUI runs:

    1. discover audio tracks   (pipeline/audio_track_discovery.py)
    2. transcription           (api/whisperx_client.py + pipeline/word_level_srt_builder.py)
    3. segment selection       (pipeline/segment_selector.py + api/manus_client.py)
    4. cutting + portrait       (pipeline/video_cutter.py, audio_cutter.py, face_cropper.py,
                                 silence_remover.py, bottom_overlay.py)

Every stage calls the same pure pipeline/ + api/ functions the screens call.
Progress is logged to stderr; the FINAL stdout line is a JSON receipt so a
wrapper script can parse the result.

Run:
    python cli.py --video PATH [options]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Windows consoles default to cp1252 and crash print()/json.dumps on diacritics (ē/ā) or
# Cyrillic. Force UTF-8 so the final receipt line never aborts an otherwise-successful run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

from models.video_segment import VideoSegment

logger = logging.getLogger("sermon_shorts.cli")

REPOSITORY_ROOT = Path(__file__).resolve().parent
POSE_MODEL_DIRECTORY = REPOSITORY_ROOT / ".models"


# ── Logging + environment ────────────────────────────────────────────────────

def configure_stderr_logging() -> None:
    """Send all pipeline progress to stderr, keeping stdout clean for the receipt."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_environment() -> None:
    """Load .env from the repository root, exactly as main.py does."""
    load_dotenv(REPOSITORY_ROOT / ".env")


def log_progress(message: str) -> None:
    """on_progress callback handed to the pipeline functions."""
    logger.info(message)


# ── Stage 1: discovery ───────────────────────────────────────────────────────

def discover_tracks(video_file_path: Path) -> dict[str, Path]:
    from pipeline.audio_track_discovery import (
        discover_audio_tracks,
        format_discovered_tracks_summary,
    )
    from pipeline.language_detect import detect_language_from_filename

    tracks = discover_audio_tracks(video_file_path)
    if not tracks:
        # Fallback: treat the video's own audio? No -- the TUI requires tracks.
        # A single sibling audio file is not auto-picked, so fail loudly instead.
        raise SystemExit(
            f"No audio tracks found in '{video_file_path.parent / 'Audio RAW'}'. "
            "Expected _A03/_A04/_A05 suffixed files."
        )
    logger.info(format_discovered_tracks_summary(tracks))
    return tracks


# ── Stage 2: transcription ───────────────────────────────────────────────────

def save_single_language_srt(
    video_file_path: Path,
    version_directory: Path,
    audio_stem: str,
    response_data: dict,
):
    """Build a word-level SRT for one track, saving to whisperx/v{N}/ and transcriptions/."""
    from pipeline.project_paths import transcriptions_directory
    from pipeline.word_level_srt_builder import (
        build_word_level_subtitles,
        save_word_level_srt,
    )

    subtitles = build_word_level_subtitles(response_data)
    save_word_level_srt(subtitles, version_directory / f"{audio_stem}_wordlevel.srt")

    compatibility_path = (
        transcriptions_directory(video_file_path, create=True)
        / f"{audio_stem}_wordlevel.srt"
    )
    save_word_level_srt(subtitles, compatibility_path)
    return compatibility_path


def merge_bilingual_subtitles(
    video_file_path: Path,
    discovered_tracks: dict[str, Path],
    version_directory: Path,
    responses_by_language: dict[str, dict],
) -> Optional[Path]:
    """
    Merge LV + RU word-level responses into <video_stem>_bilingual_wordlevel.srt.
    Returns the transcriptions/ copy path, or None if both languages are not present.
    """
    if "lv" not in responses_by_language or "ru" not in responses_by_language:
        return None

    from pipeline.audio_compressor import DEFAULT_AUDIO_OFFSET_SECONDS
    from pipeline.project_paths import transcriptions_directory
    from pipeline.word_level_srt_builder import (
        build_cleaned_speech_regions,
        extract_words_from_response,
        map_words_to_speech_regions,
        merge_bilingual_word_level_srt,
        save_word_level_srt,
    )

    russian_speech, latvian_speech = build_cleaned_speech_regions(
        primary_audio_path=discovered_tracks["ru"],
        secondary_audio_path=discovered_tracks["lv"],
        offset_seconds=DEFAULT_AUDIO_OFFSET_SECONDS,
        on_progress=log_progress,
    )

    russian_subtitles = map_words_to_speech_regions(
        extract_words_from_response(responses_by_language["ru"]), russian_speech, "RU",
    )
    latvian_subtitles = map_words_to_speech_regions(
        extract_words_from_response(responses_by_language["lv"]), latvian_speech, "LV",
    )

    merged_subtitles = merge_bilingual_word_level_srt(russian_subtitles, latvian_subtitles)

    video_stem = video_file_path.stem
    save_word_level_srt(
        merged_subtitles, version_directory / f"{video_stem}_bilingual_wordlevel.srt",
    )
    compatibility_path = (
        transcriptions_directory(video_file_path, create=True)
        / f"{video_stem}_bilingual_wordlevel.srt"
    )
    save_word_level_srt(merged_subtitles, compatibility_path)
    logger.info("Bilingual SRT built: %s (%d entries)", compatibility_path.name, len(merged_subtitles))
    return compatibility_path


def transcribe_tracks_fresh(
    video_file_path: Path,
    discovered_tracks: dict[str, Path],
) -> tuple[Path, Optional[Path]]:
    """
    Upload every track to WhisperX, store versioned JSON + word-level SRT, and
    merge LV+RU into a bilingual SRT. Mirrors TranscriptionScreen._transcribe_all_audio_tracks.
    Returns (primary_subtitle_path, bilingual_srt_path_or_None).
    """
    from api.whisperx_client import WhisperXClient
    from pipeline.audio_compressor import prepare_audio_with_offset
    from pipeline.language_detect import language_name
    from pipeline.whisperx_response_storage import (
        build_next_version_directory_path,
        save_whisperx_response,
    )

    whisperx_api_url = os.getenv("WHISPERX_API_URL", "") or "https://wsp.kingdom.lv"
    whisperx_api_key = os.getenv("WSP_API_KEY", "") or os.getenv("WHISPERX_API_KEY", "")
    if not whisperx_api_url or not whisperx_api_key:
        raise SystemExit("WSP_API_KEY (or WHISPERX_API_KEY) not set in environment / .env")

    client = WhisperXClient(api_base_url=whisperx_api_url, api_key=whisperx_api_key)
    version_directory = build_next_version_directory_path(video_file_path)
    logger.info("Storing WhisperX results in: %s/", version_directory.name)

    responses_by_language: dict[str, dict] = {}
    primary_subtitle_path: Optional[Path] = None

    for language_code, audio_file_path in discovered_tracks.items():
        logger.info("Transcribing %s (%s)", language_name(language_code), audio_file_path.name)
        prepared_audio_path = prepare_audio_with_offset(audio_file_path, on_progress=log_progress)
        raw_response = client.transcribe_audio_file_raw(
            audio_file_path=prepared_audio_path,
            language_code=language_code,
            on_progress=log_progress,
        )
        save_whisperx_response(
            video_file_path=video_file_path,
            audio_file_stem=audio_file_path.stem,
            response_data=raw_response,
            version_directory=version_directory,
        )
        responses_by_language[language_code] = raw_response
        compatibility_path = save_single_language_srt(
            video_file_path, version_directory, audio_file_path.stem, raw_response,
        )
        if language_code == "lv" or primary_subtitle_path is None:
            primary_subtitle_path = compatibility_path

    bilingual_path = merge_bilingual_subtitles(
        video_file_path, discovered_tracks, version_directory, responses_by_language,
    )
    return (bilingual_path or primary_subtitle_path), bilingual_path


def load_transcription_existing(
    video_file_path: Path,
    discovered_tracks: dict[str, Path],
    version_number: int,
) -> tuple[Path, Optional[Path]]:
    """
    Rebuild SRTs from already-stored WhisperX JSON (whisperx/v{N}/).
    Mirrors TranscriptionScreen._load_existing_whisperx_data.
    Returns (primary_subtitle_path, bilingual_srt_path_or_None).
    """
    from pipeline.whisperx_response_storage import (
        build_version_directory_path,
        list_stored_audio_stems_for_version,
        load_whisperx_response,
    )

    version_directory = build_version_directory_path(video_file_path, version_number)
    stored_stems = list_stored_audio_stems_for_version(video_file_path, version_number)
    if not stored_stems:
        raise SystemExit(f"No stored WhisperX responses found in v{version_number}/")

    logger.info("Reusing WhisperX v%d: %s", version_number, ", ".join(stored_stems))

    responses_by_language: dict[str, dict] = {}
    primary_subtitle_path: Optional[Path] = None

    for audio_stem in stored_stems:
        response_data = load_whisperx_response(video_file_path, audio_stem, version_number)
        compatibility_path = save_single_language_srt(
            video_file_path, version_directory, audio_stem, response_data,
        )
        if "_A03" in audio_stem:
            responses_by_language["lv"] = response_data
        elif "_A04" in audio_stem:
            responses_by_language["ru"] = response_data
        if "_A03" in audio_stem or primary_subtitle_path is None:
            primary_subtitle_path = compatibility_path

    bilingual_path = merge_bilingual_subtitles(
        video_file_path, discovered_tracks, version_directory, responses_by_language,
    )
    return (bilingual_path or primary_subtitle_path), bilingual_path


def ensure_transcription(
    video_file_path: Path,
    discovered_tracks: dict[str, Path],
    reuse: bool,
) -> tuple[Path, Optional[Path]]:
    """Reuse stored WhisperX data when --reuse and a version exists; otherwise transcribe fresh."""
    from pipeline.whisperx_response_storage import find_latest_version_number

    latest_version = find_latest_version_number(video_file_path)
    if reuse and latest_version is not None:
        return load_transcription_existing(video_file_path, discovered_tracks, latest_version)
    return transcribe_tracks_fresh(video_file_path, discovered_tracks)


# ── Stage 3: segment selection ───────────────────────────────────────────────

def save_manus_response(video_file_path: Path, response_text: str) -> Path:
    from pipeline.project_paths import data_directory

    responses_directory = data_directory(video_file_path, create=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = responses_directory / f"manus_response_{timestamp}.json"
    output_path.write_text(response_text, encoding="utf-8", errors="replace")
    logger.info("Manus response saved: %s", output_path)
    return output_path


def find_latest_cached_manus_response(video_file_path: Path) -> Optional[Path]:
    from pipeline.project_paths import data_directory

    responses_directory = data_directory(video_file_path)
    if not responses_directory.exists():
        return None
    cached = sorted(responses_directory.glob("manus_response_*.json"), reverse=True)
    return cached[0] if cached else None


def build_srt_content_for_prompt(all_subtitles) -> str:
    return "\n".join(
        f"{subtitle.index}\n{subtitle.start} --> {subtitle.end}\n{subtitle.content}\n"
        for subtitle in all_subtitles
    )


def request_new_manus_analysis(
    video_file_path: Path,
    all_subtitles,
    minimum_clips: int,
    maximum_clips: int,
) -> str:
    from api.manus_client import ManusClient
    import pipeline.segment_selector as segment_selector

    manus_api_key = os.getenv("MANUS_API_KEY", "")
    if not manus_api_key:
        raise SystemExit("MANUS_API_KEY not set in environment / .env")

    # Honour --min-clips/--max-clips without editing the module source.
    segment_selector.MINIMUM_CLIPS = minimum_clips
    segment_selector.MAXIMUM_CLIPS = maximum_clips
    prompt = segment_selector.build_segment_selection_prompt(
        build_srt_content_for_prompt(all_subtitles)
    )

    manus_project_id = os.getenv("MANUS_PROJECT_ID", "")
    client = ManusClient(api_key=manus_api_key)
    try:
        response_text = client.submit_prompt_and_wait_for_response(
            prompt,
            on_progress=log_progress,
            project_id=manus_project_id or None,
            task_title=video_file_path.stem,
        )
    finally:
        client.close()
    return response_text


def resume_manus_task(task_id: str) -> str:
    from api.manus_client import ManusClient

    manus_api_key = os.getenv("MANUS_API_KEY", "")
    if not manus_api_key:
        raise SystemExit("MANUS_API_KEY not set in environment / .env")

    client = ManusClient(api_key=manus_api_key)
    try:
        return client.fetch_task_response(task_id, on_progress=log_progress)
    finally:
        client.close()


def select_segments(
    video_file_path: Path,
    subtitle_file_path: Path,
    minimum_clips: int,
    maximum_clips: int,
    reuse: bool,
    manus_task_id: Optional[str],
) -> tuple[list[VideoSegment], Optional[Path]]:
    """
    Obtain viral segments from Manus. Priority:
        1. --manus-task-id  -> fetch that task
        2. --reuse + cached manus_response_*.json -> reuse newest
        3. otherwise submit a fresh prompt and poll
    Returns (segments, manus_response_json_path_or_None).
    """
    from pipeline.segment_selector import parse_segments_from_manus_response
    from pipeline.subtitle_parser import load_subtitle_file

    all_subtitles = load_subtitle_file(subtitle_file_path)
    logger.info("Loaded %d subtitle entries for segment selection", len(all_subtitles))

    if manus_task_id:
        logger.info("Resuming Manus task: %s", manus_task_id)
        response_text = resume_manus_task(manus_task_id)
        manus_response_path = save_manus_response(video_file_path, response_text)
    else:
        cached_response_path = find_latest_cached_manus_response(video_file_path) if reuse else None
        if cached_response_path is not None:
            logger.info("Reusing cached Manus response: %s", cached_response_path.name)
            response_text = cached_response_path.read_text(encoding="utf-8", errors="replace")
            manus_response_path = cached_response_path
        else:
            response_text = request_new_manus_analysis(
                video_file_path, all_subtitles, minimum_clips, maximum_clips,
            )
            manus_response_path = save_manus_response(video_file_path, response_text)

    segments = parse_segments_from_manus_response(response_text, all_subtitles)
    logger.info("Manus selected %d segments", len(segments))
    return segments, manus_response_path


# ── Stage 4: cutting + portrait rendering ────────────────────────────────────

def prepare_offset_audio_for_cutting(discovered_tracks: dict[str, Path]) -> dict[str, Path]:
    from pipeline.audio_compressor import prepare_original_audio_with_offset
    from pipeline.language_detect import language_name

    offset_audio_by_language: dict[str, Path] = {}
    for language_code, audio_file_path in discovered_tracks.items():
        offset_path = prepare_original_audio_with_offset(audio_file_path, on_progress=log_progress)
        offset_audio_by_language[language_code] = offset_path
        logger.info("%s offset audio ready: %s", language_name(language_code), offset_path.name)
    return offset_audio_by_language


def snap_segment_start_to_subtitle_boundary(segment: VideoSegment, all_subtitles) -> None:
    """Move the cut start back to the end of the preceding subtitle (matches cutting_screen)."""
    from pipeline.subtitle_parser import find_end_of_preceding_subtitle

    preceding_end = find_end_of_preceding_subtitle(all_subtitles, segment.start_time)
    if preceding_end is not None and preceding_end != segment.start_time:
        original_timestamp = segment.ffmpeg_start_timestamp
        segment.start_time = preceding_end
        logger.info(
            "Segment %d: snapped start %s -> %s (subtitle boundary)",
            segment.index, original_timestamp, segment.ffmpeg_start_timestamp,
        )


def cut_one_segment(
    video_file_path: Path,
    segment: VideoSegment,
    clips_directory: Path,
    all_subtitles,
    offset_audio_by_language: dict[str, Path],
) -> dict:
    """Cut one landscape clip + its SRT, .md description, and per-language WAVs."""
    from pipeline.audio_cutter import build_output_audio_file_path, cut_audio_segment
    from pipeline.clip_description import write_clip_description_file
    from pipeline.subtitle_parser import (
        save_subtitles_to_file,
        slice_subtitles_within_window,
    )
    from pipeline.video_cutter import (
        build_output_subtitle_file_path,
        build_output_video_file_path,
        cut_segment_from_video,
    )

    snap_segment_start_to_subtitle_boundary(segment, all_subtitles)

    video_output_path = build_output_video_file_path(clips_directory, segment, video_file_path)
    logger.info(
        "Cutting segment %d: %s -> %s",
        segment.index, segment.ffmpeg_start_timestamp, segment.ffmpeg_end_timestamp,
    )
    cut_segment_from_video(
        source_video_path=video_file_path,
        segment=segment,
        output_file_path=video_output_path,
    )

    subtitle_output_path = build_output_subtitle_file_path(clips_directory, segment, video_file_path)
    sliced = slice_subtitles_within_window(
        all_subtitles, window_start=segment.start_time, window_end=segment.end_time,
    )
    save_subtitles_to_file(sliced, subtitle_output_path)

    description_path = write_clip_description_file(segment, video_output_path)

    for language_code, offset_audio_path in offset_audio_by_language.items():
        audio_output_path = build_output_audio_file_path(
            clips_directory, segment, language_code, video_file_path,
        )
        cut_audio_segment(
            source_audio_path=offset_audio_path,
            segment=segment,
            output_file_path=audio_output_path,
        )

    return {
        "segment": segment,
        "mp4": video_output_path,
        "srt": subtitle_output_path,
        "md": description_path,
        "portraitMp4": None,
    }


def render_one_portrait(
    landscape_video_path: Path,
    segment: VideoSegment,
    remove_silence: bool,
    speed_multiplier: float,
    overlay_height: int,
    minimum_silence_seconds: float,
) -> Path:
    """
    Portrait 9:16 crop -> optional silence removal -> bottom overlay.
    Returns the deepest deliverable that was produced (square > trimmed > portrait).
    """
    from pipeline.bottom_overlay import (
        apply_bottom_overlay,
        build_square_output_path,
        derive_landscape_clip_path,
    )
    from pipeline.face_cropper import build_portrait_output_path, crop_segment_to_portrait
    from pipeline.silence_remover import (
        SilenceRemovalError,
        build_silence_removed_output_path,
        remove_silence_from_video,
    )

    portrait_output_path = build_portrait_output_path(landscape_video_path)
    logger.info("Segment %d: cropping to portrait...", segment.index)
    crop_segment_to_portrait(
        source_video_path=landscape_video_path,
        output_file_path=portrait_output_path,
        start_seconds=0.0,
        duration_seconds=segment.duration_seconds,
        model_directory=POSE_MODEL_DIRECTORY,
        speed_multiplier=speed_multiplier,
        on_progress=None,
    )

    overlay_input_path = portrait_output_path
    if remove_silence:
        trimmed_output_path = build_silence_removed_output_path(portrait_output_path)
        try:
            overlay_input_path = remove_silence_from_video(
                source_video_path=portrait_output_path,
                output_file_path=trimmed_output_path,
                minimum_duration_seconds=minimum_silence_seconds,
                on_progress=log_progress,
            )
        except SilenceRemovalError as error:
            logger.error("Segment %d silence removal failed: %s", segment.index, error)
            overlay_input_path = portrait_output_path

    square_output_path = build_square_output_path(overlay_input_path)
    apply_bottom_overlay(
        source_video_path=overlay_input_path,
        output_file_path=square_output_path,
        landscape_video_path=derive_landscape_clip_path(overlay_input_path),
        overlay_height=overlay_height,
        on_progress=log_progress,
    )

    for candidate in (square_output_path, overlay_input_path, portrait_output_path):
        if candidate.exists():
            return candidate
    return portrait_output_path


def cut_all_segments(
    video_file_path: Path,
    subtitle_file_path: Path,
    segments: list[VideoSegment],
    clips_directory: Path,
    discovered_tracks: dict[str, Path],
    enable_portrait: bool,
    remove_silence: bool,
    speed_multiplier: float,
    overlay_height: int,
    minimum_silence_seconds: float,
) -> list[dict]:
    """Cut every segment, then optionally render portrait versions. Mirrors CuttingScreen."""
    from pipeline.subtitle_parser import load_subtitle_file

    clips_directory.mkdir(parents=True, exist_ok=True)
    all_subtitles = load_subtitle_file(subtitle_file_path)
    offset_audio_by_language = prepare_offset_audio_for_cutting(discovered_tracks)

    clip_records: list[dict] = []
    for segment in segments:
        try:
            record = cut_one_segment(
                video_file_path, segment, clips_directory, all_subtitles, offset_audio_by_language,
            )
            clip_records.append(record)
            logger.info("Segment %d saved: %s", segment.index, record["mp4"].name)
        except Exception as error:
            logger.error("Segment %d failed: %s", segment.index, error)

    if enable_portrait:
        for record in clip_records:
            try:
                record["portraitMp4"] = render_one_portrait(
                    landscape_video_path=record["mp4"],
                    segment=record["segment"],
                    remove_silence=remove_silence,
                    speed_multiplier=speed_multiplier,
                    overlay_height=overlay_height,
                    minimum_silence_seconds=minimum_silence_seconds,
                )
                logger.info("Segment %d portrait: %s", record["segment"].index, record["portraitMp4"].name)
            except Exception as error:
                logger.error("Segment %d portrait render failed: %s", record["segment"].index, error)

    return clip_records


# ── Receipt assembly ─────────────────────────────────────────────────────────

def absolute_or_none(path: Optional[Path]) -> Optional[str]:
    return str(path.resolve()) if path is not None else None


def build_clip_entry(segment: VideoSegment, record: Optional[dict]) -> dict:
    return {
        "title": segment.suggested_title,
        "startSec": segment.start_seconds,
        "endSec": segment.end_seconds,
        "viralScore": segment.overall_viral_score,
        "mp4": absolute_or_none(record["mp4"]) if record else None,
        "portraitMp4": absolute_or_none(record["portraitMp4"]) if record else None,
        "srt": absolute_or_none(record["srt"]) if record else None,
        "md": absolute_or_none(record["md"]) if record else None,
    }


def build_receipt(
    status: str,
    video_file_path: Path,
    bilingual_srt_path: Optional[Path],
    manus_response_path: Optional[Path],
    clips_directory: Path,
    segments: list[VideoSegment],
    clip_records: list[dict],
) -> dict:
    from pipeline.project_paths import event_root_directory

    records_by_segment = {id(record["segment"]): record for record in clip_records}
    clips = [
        build_clip_entry(segment, records_by_segment.get(id(segment)))
        for segment in segments
    ]
    return {
        "status": status,
        "video": str(video_file_path.resolve()),
        "eventRoot": str(event_root_directory(video_file_path).resolve()),
        "bilingualSrt": absolute_or_none(bilingual_srt_path),
        "manusResponse": absolute_or_none(manus_response_path),
        "numSegments": len(segments),
        "clipsDir": str(clips_directory.resolve()),
        "clips": clips,
    }


def emit_receipt(receipt: dict, json_output_path: Optional[Path]) -> None:
    receipt_text = json.dumps(receipt, ensure_ascii=False, indent=2)
    if json_output_path is not None:
        json_output_path.parent.mkdir(parents=True, exist_ok=True)
        json_output_path.write_text(receipt_text, encoding="utf-8")
        logger.info("Receipt written to %s", json_output_path)
    # FINAL stdout line: single-line JSON so a wrapper can parse it deterministically.
    print(json.dumps(receipt, ensure_ascii=False))


# ── Argument parsing + main ──────────────────────────────────────────────────

def parse_arguments(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Headless Sermon Shorts viral-clip pipeline driver.",
    )
    parser.add_argument("--video", required=True, type=Path, help="source video (required)")
    parser.add_argument("--clips-dir", type=Path, default=None,
                        help="output dir (default: <event root>/Clips RAW)")
    parser.add_argument("--min-clips", type=int, default=5)
    parser.add_argument("--max-clips", type=int, default=14)
    parser.add_argument("--portrait", action=argparse.BooleanOptionalAction, default=True,
                        help="portrait 9:16 face crop (default: on)")
    parser.add_argument("--silence-remove", dest="silence_remove",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="remove silences from portrait clips (default: on)")
    parser.add_argument("--speed", type=float, default=1.3, help="portrait speed multiplier")
    parser.add_argument("--overlay-height", type=int, default=695)
    parser.add_argument("--min-silence", type=float, default=1.0)
    parser.add_argument("--reuse", action="store_true",
                        help="reuse existing whisperx transcript + cached manus response if present")
    parser.add_argument("--manus-task-id", type=str, default=None,
                        help="resume a specific Manus task id instead of submitting a new one")
    parser.add_argument("--dry-run", action="store_true",
                        help="discovery + transcription + Manus selection only; print segments, do not cut")
    parser.add_argument("--json", dest="json_path", type=Path, default=None,
                        help="write a JSON receipt to this path")
    return parser.parse_args(argv)


def run_pipeline(arguments: argparse.Namespace) -> dict:
    from pipeline.project_paths import clips_directory as default_clips_directory

    video_file_path = arguments.video.resolve()
    if not video_file_path.is_file():
        raise SystemExit(f"Video file not found: {video_file_path}")

    clips_directory = (
        arguments.clips_dir.resolve()
        if arguments.clips_dir is not None
        else default_clips_directory(video_file_path)
    )

    discovered_tracks = discover_tracks(video_file_path)

    subtitle_file_path, bilingual_srt_path = ensure_transcription(
        video_file_path, discovered_tracks, arguments.reuse,
    )
    if subtitle_file_path is None or not subtitle_file_path.exists():
        raise SystemExit("Transcription produced no usable subtitle file")

    segments, manus_response_path = select_segments(
        video_file_path,
        subtitle_file_path,
        arguments.min_clips,
        arguments.max_clips,
        arguments.reuse,
        arguments.manus_task_id,
    )
    if not segments:
        raise SystemExit("Manus returned zero segments")

    if arguments.dry_run:
        for segment in segments:
            logger.info(
                "DRY-RUN segment %d: %s  [%.1fs -> %.1fs]  score=%.1f",
                segment.index, segment.suggested_title,
                segment.start_seconds, segment.end_seconds, segment.overall_viral_score,
            )
        return build_receipt(
            "dry-run", video_file_path, bilingual_srt_path, manus_response_path,
            clips_directory, segments, clip_records=[],
        )

    clip_records = cut_all_segments(
        video_file_path=video_file_path,
        subtitle_file_path=subtitle_file_path,
        segments=segments,
        clips_directory=clips_directory,
        discovered_tracks=discovered_tracks,
        enable_portrait=arguments.portrait,
        remove_silence=arguments.silence_remove,
        speed_multiplier=arguments.speed,
        overlay_height=arguments.overlay_height,
        minimum_silence_seconds=arguments.min_silence,
    )
    return build_receipt(
        "ok", video_file_path, bilingual_srt_path, manus_response_path,
        clips_directory, segments, clip_records,
    )


def main(argv: Optional[list[str]] = None) -> int:
    configure_stderr_logging()
    load_environment()
    arguments = parse_arguments(argv)

    try:
        receipt = run_pipeline(arguments)
    except SystemExit as error:
        message = str(error.code) if error.code not in (None, 0) else ""
        if message:
            logger.error(message)
        print(json.dumps({"status": "error", "message": message}, ensure_ascii=False))
        return 1
    except Exception as error:
        logger.exception("Pipeline failed")
        print(json.dumps({"status": "error", "message": f"{type(error).__name__}: {error}"},
                         ensure_ascii=False))
        return 1

    emit_receipt(receipt, arguments.json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
