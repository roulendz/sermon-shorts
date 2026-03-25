"""
pipeline/word_level_srt_builder.py

Builds SRT subtitle files from WhisperX word-level timestamp data.
Uses word-level start/end times for precise subtitle boundaries
while keeping segment text as complete sentences.

For bilingual merge: maps individual WhisperX words into speech regions
detected by silence analysis, then builds one subtitle per region.
This ensures clean RU-LV alternation without overlap.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import srt

from pipeline.subtitle_parser import serialize_subtitles_to_srt_text

logger = logging.getLogger(__name__)


def build_word_level_subtitles(
    whisperx_response: dict[str, Any],
) -> list[srt.Subtitle]:
    """
    Convert WhisperX response data into SRT subtitles using word-level
    timestamps for precise segment boundaries.

    Each WhisperX segment becomes one SRT entry. The start time is
    taken from the first word's start, and the end time from the last
    word's end. The text is the full segment text.

    Falls back to segment-level timestamps if word data is missing.
    """
    segments = _extract_segments_from_response(whisperx_response)
    subtitles: list[srt.Subtitle] = []

    for subtitle_index, segment in enumerate(segments, start=1):
        segment_text = segment.get("text", "").strip()
        if not segment_text:
            continue

        words = segment.get("words", [])

        if words:
            word_level_start = _find_first_valid_word_start(words)
            word_level_end = _find_last_valid_word_end(words)
            start_seconds = (
                word_level_start
                if word_level_start is not None
                else segment.get("start", 0)
            )
            end_seconds = (
                word_level_end
                if word_level_end is not None
                else segment.get("end", 0)
            )
        else:
            start_seconds = segment.get("start", 0)
            end_seconds = segment.get("end", 0)

        if end_seconds <= start_seconds:
            continue

        subtitles.append(
            srt.Subtitle(
                index=subtitle_index,
                start=timedelta(seconds=start_seconds),
                end=timedelta(seconds=end_seconds),
                content=segment_text,
            )
        )

    logger.info(f"Built {len(subtitles)} word-level subtitles from WhisperX data")
    return subtitles


def save_word_level_srt(
    subtitles: list[srt.Subtitle],
    output_file_path: Path,
) -> Path:
    """Save word-level subtitles to an SRT file."""
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    srt_text = serialize_subtitles_to_srt_text(subtitles)
    output_file_path.write_text(srt_text, encoding="utf-8")
    logger.info(f"Word-level SRT saved: {output_file_path}")
    return output_file_path


def _extract_segments_from_response(
    whisperx_response: dict[str, Any],
) -> list[dict]:
    """
    Extract the segments list from a WhisperX response.
    Handles both direct segment lists and nested result structures.
    """
    if isinstance(whisperx_response, list):
        return whisperx_response

    result = whisperx_response.get("result", whisperx_response)
    if isinstance(result, dict):
        return result.get("segments", [])
    if isinstance(result, list):
        return result
    return []


def _find_first_valid_word_start(words: list[dict]) -> float | None:
    """Find the start time of the first word that has a valid start timestamp."""
    for word in words:
        start = word.get("start")
        if start is not None:
            return float(start)
    return None


def build_cleaned_speech_regions(
    primary_audio_path: Path,
    secondary_audio_path: Path,
    offset_seconds: float = 0.0,
    noise_threshold_db: int = -25,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Detect speech regions for both tracks, then clean them using the
    full bilingual algorithm: mic-bleed removal, RU-priority alignment,
    minimum duration filter. Returns (primary_speech, secondary_speech)
    with offset applied.

    Reuses the proven functions from bilingual_subtitle_merger.
    """
    from pipeline.silence_detection import (
        detect_silence_regions,
        silence_regions_to_speech_regions,
        get_audio_duration_seconds,
    )
    from pipeline.bilingual_subtitle_merger import (
        _remove_mic_bleed,
        _align_secondary_to_primary,
    )

    def _log(message: str) -> None:
        if on_progress:
            on_progress(message)

    # Primary (RU) speech regions
    primary_duration = get_audio_duration_seconds(str(primary_audio_path))
    primary_silences = detect_silence_regions(
        str(primary_audio_path), noise_threshold_db, on_progress=on_progress,
    )
    primary_speech_raw = silence_regions_to_speech_regions(primary_silences, primary_duration)
    primary_speech = [(start, end) for start, end in primary_speech_raw if end - start >= 0.5]
    _log(f"Primary: {len(primary_speech_raw)} raw -> {len(primary_speech)} regions (>=0.5s)")

    # Secondary (LV) speech regions
    secondary_duration = get_audio_duration_seconds(str(secondary_audio_path))
    secondary_silences = detect_silence_regions(
        str(secondary_audio_path), noise_threshold_db, on_progress=on_progress,
    )
    secondary_speech_raw = silence_regions_to_speech_regions(secondary_silences, secondary_duration)
    secondary_speech = [(start, end) for start, end in secondary_speech_raw if end - start >= 0.5]
    _log(f"Secondary: {len(secondary_speech_raw)} raw -> {len(secondary_speech)} regions (>=0.5s)")

    # Remove mic bleed on secondary (LV) — RU is priority, never removed
    secondary_speech = _remove_mic_bleed(secondary_speech, primary_speech)
    _log(f"Secondary after bleed removal: {len(secondary_speech)} regions")

    # Align secondary to primary (trim LV when RU is talking)
    secondary_speech = _align_secondary_to_primary(primary_speech, secondary_speech)
    _log(f"Secondary after RU-priority alignment: {len(secondary_speech)} regions")

    # Apply video offset
    primary_speech_offset = [
        (start + offset_seconds, end + offset_seconds) for start, end in primary_speech
    ]
    secondary_speech_offset = [
        (start + offset_seconds, end + offset_seconds) for start, end in secondary_speech
    ]

    return primary_speech_offset, secondary_speech_offset


def extract_words_from_response(
    whisperx_response: dict[str, Any],
) -> list[dict]:
    """
    Extract all individual words with timestamps from a WhisperX response.
    Returns a flat list of word dicts, each with 'word', 'start', 'end'.
    Words without timestamps are skipped.
    """
    segments = _extract_segments_from_response(whisperx_response)
    all_words: list[dict] = []

    for segment in segments:
        words = segment.get("words", [])
        for word in words:
            if word.get("start") is not None and word.get("end") is not None:
                all_words.append(word)

    logger.info(f"Extracted {len(all_words)} timestamped words from WhisperX response")
    return all_words


def map_words_to_speech_regions(
    words: list[dict],
    speech_regions: list[tuple[float, float]],
    language_tag: str,
) -> list[srt.Subtitle]:
    """
    Map individual WhisperX words into speech region slots. Each speech
    region becomes one subtitle entry. Words are placed into a region
    when their start timestamp falls within it.

    Uses REGION boundaries for subtitle timing (not word timestamps).
    This guarantees zero overlaps since cleaned speech regions are
    non-overlapping after mic-bleed removal and RU-priority alignment.

    Regions with no words are skipped (silence or mic bleed).
    """
    subtitles: list[srt.Subtitle] = []

    for region_start, region_end in speech_regions:
        region_text_parts: list[str] = []

        for word in words:
            word_start = word["start"]
            if region_start <= word_start <= region_end:
                region_text_parts.append(word.get("word", ""))

        text = " ".join(region_text_parts).strip()
        if not text:
            continue

        subtitles.append(
            srt.Subtitle(
                index=0,
                start=timedelta(seconds=region_start),
                end=timedelta(seconds=region_end),
                content=f"[{language_tag}] {text}",
            )
        )

    logger.info(
        f"Mapped {len(words)} words into {len(subtitles)} subtitles "
        f"across {len(speech_regions)} speech regions [{language_tag}]"
    )
    return subtitles


def merge_bilingual_word_level_srt(
    primary_subtitles: list[srt.Subtitle],
    secondary_subtitles: list[srt.Subtitle],
) -> list[srt.Subtitle]:
    """
    Merge two pre-tagged subtitle tracks into one bilingual SRT sorted
    by timestamp. Subtitles must already have [RU]/[LV] tags and be
    mapped to non-overlapping speech regions.
    """
    all_entries = primary_subtitles + secondary_subtitles
    all_entries.sort(key=lambda subtitle: subtitle.start)

    for new_index, subtitle in enumerate(all_entries, start=1):
        subtitle.index = new_index

    logger.info(
        f"Merged bilingual SRT: {len(primary_subtitles)} primary "
        f"+ {len(secondary_subtitles)} secondary "
        f"= {len(all_entries)} total entries"
    )
    return all_entries


def _find_last_valid_word_end(words: list[dict]) -> float | None:
    """Find the end time of the last word that has a valid end timestamp."""
    for word in reversed(words):
        end = word.get("end")
        if end is not None:
            return float(end)
    return None
