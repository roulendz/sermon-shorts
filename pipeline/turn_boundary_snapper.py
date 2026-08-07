"""
pipeline/turn_boundary_snapper.py

Snaps Manus-selected segment boundaries to consecutive-translation
turn pairs so a clip always opens with the original [RU] passage and
closes after its [LV] translation — never mid-pair, never mid-sentence.

The sermon alternates RU (pastor, original) -> LV (translator, echo).
A "translation pair" is one RU speaking turn plus the LV turn that
follows it. Clip boundaries chosen by the AI are snapped outward:
start moves back to the pair's RU turn start, end moves forward to
the pair's LV turn end.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Sequence

import srt

from models.video_segment import VideoSegment

logger = logging.getLogger(__name__)

LANGUAGE_TAG_PATTERN = re.compile(r"^\[([A-Z]{2})\]")

PRIMARY_LANGUAGE_TAG = "RU"

MINIMUM_CLIP_DURATION = timedelta(seconds=45)
MAXIMUM_CLIP_DURATION = timedelta(seconds=105)

BOUNDARY_PADDING = timedelta(milliseconds=300)


@dataclass(frozen=True)
class SpeakingTurn:
    """A contiguous run of same-language subtitle cues."""
    language_tag: str
    start: timedelta
    end: timedelta


@dataclass(frozen=True)
class TranslationPair:
    """One original (RU) turn plus its translation (LV) turn, if present."""
    original_turn: Optional[SpeakingTurn]
    translation_turn: Optional[SpeakingTurn]

    @property
    def start(self) -> timedelta:
        first_turn = self.original_turn or self.translation_turn
        return first_turn.start

    @property
    def end(self) -> timedelta:
        last_turn = self.translation_turn or self.original_turn
        return last_turn.end


def extract_language_tag(subtitle_content: str) -> Optional[str]:
    match = LANGUAGE_TAG_PATTERN.match(subtitle_content.strip())
    return match.group(1) if match else None


def build_speaking_turns(subtitles: Sequence[srt.Subtitle]) -> list[SpeakingTurn]:
    """Group consecutive same-language cues into speaking turns.

    Returns an empty list when the subtitles carry no language tags
    (single-language sermons) — snapping is then a no-op.
    """
    turns: list[SpeakingTurn] = []
    current_tag: Optional[str] = None
    current_start: Optional[timedelta] = None
    current_end: Optional[timedelta] = None

    for subtitle in subtitles:
        language_tag = extract_language_tag(subtitle.content)
        if language_tag is None:
            continue
        if language_tag == current_tag:
            current_end = max(current_end, subtitle.end)
        else:
            if current_tag is not None:
                turns.append(SpeakingTurn(current_tag, current_start, current_end))
            current_tag = language_tag
            current_start = subtitle.start
            current_end = subtitle.end

    if current_tag is not None:
        turns.append(SpeakingTurn(current_tag, current_start, current_end))

    return turns


def build_translation_pairs(turns: Sequence[SpeakingTurn]) -> list[TranslationPair]:
    """Pair each primary-language (RU) turn with the turn that follows it.

    A leading non-primary turn (translation of something before the
    recording window) forms a translation-only pair.
    """
    pairs: list[TranslationPair] = []
    pending_original: Optional[SpeakingTurn] = None

    for turn in turns:
        if turn.language_tag == PRIMARY_LANGUAGE_TAG:
            if pending_original is not None:
                pairs.append(TranslationPair(pending_original, None))
            pending_original = turn
        else:
            pairs.append(TranslationPair(pending_original, turn))
            pending_original = None

    if pending_original is not None:
        pairs.append(TranslationPair(pending_original, None))

    return pairs


def snap_time_window_to_pairs(
    window_start: timedelta,
    window_end: timedelta,
    pairs: Sequence[TranslationPair],
    minimum_duration: timedelta = MINIMUM_CLIP_DURATION,
    maximum_duration: timedelta = MAXIMUM_CLIP_DURATION,
) -> tuple[timedelta, timedelta]:
    """Snap [window_start, window_end] outward to whole translation pairs.

    Start snaps back to the start of the pair containing window_start;
    end snaps forward to the end of the pair containing window_end.
    The pair range is then trimmed (drop trailing pairs) if it exceeds
    maximum_duration, or extended forward if below minimum_duration.
    """
    if not pairs:
        return window_start, window_end

    first_pair_index = _find_pair_index_containing(window_start, pairs)
    last_pair_index = _find_pair_index_containing(
        window_end, pairs, prefer_preceding=True,
    )
    if last_pair_index < first_pair_index:
        last_pair_index = first_pair_index

    def duration_of_range(start_index: int, end_index: int) -> timedelta:
        return pairs[end_index].end - pairs[start_index].start

    while (
        duration_of_range(first_pair_index, last_pair_index) > maximum_duration
        and last_pair_index > first_pair_index
    ):
        last_pair_index -= 1

    while (
        duration_of_range(first_pair_index, last_pair_index) < minimum_duration
        and last_pair_index + 1 < len(pairs)
    ):
        last_pair_index += 1
    # Extending forward may overshoot; if it did and there is room, back off.
    if (
        duration_of_range(first_pair_index, last_pair_index) > maximum_duration
        and last_pair_index > first_pair_index
    ):
        last_pair_index -= 1

    snapped_start = pairs[first_pair_index].start - BOUNDARY_PADDING
    if first_pair_index > 0:
        snapped_start = max(snapped_start, pairs[first_pair_index - 1].end)
    snapped_start = max(snapped_start, timedelta(0))

    snapped_end = pairs[last_pair_index].end + BOUNDARY_PADDING
    if last_pair_index + 1 < len(pairs):
        snapped_end = min(snapped_end, pairs[last_pair_index + 1].start)

    return snapped_start, snapped_end


def snap_segments_to_translation_pairs(
    segments: Sequence[VideoSegment],
    all_subtitles: Sequence[srt.Subtitle],
) -> None:
    """Snap every segment's boundaries to whole RU->LV translation pairs.

    Mutates segments in place. No-op when the subtitles carry no
    language tags (single-language sermon).
    """
    turns = build_speaking_turns(all_subtitles)
    if not turns:
        logger.info("No language tags in subtitles — turn snapping skipped")
        return

    pairs = build_translation_pairs(turns)
    logger.info(
        "Turn snapping: %d turns -> %d translation pairs", len(turns), len(pairs),
    )

    for segment in segments:
        snapped_start, snapped_end = snap_time_window_to_pairs(
            segment.start_time, segment.end_time, pairs,
        )
        if snapped_start == segment.start_time and snapped_end == segment.end_time:
            continue
        logger.info(
            "Segment %d: snapped %s-%s -> %s-%s (translation pairs)",
            segment.index,
            _format_seconds(segment.start_time), _format_seconds(segment.end_time),
            _format_seconds(snapped_start), _format_seconds(snapped_end),
        )
        segment.start_time = snapped_start
        segment.end_time = snapped_end


def _find_pair_index_containing(
    reference_time: timedelta,
    pairs: Sequence[TranslationPair],
    prefer_preceding: bool = False,
) -> int:
    """Index of the pair containing reference_time.

    When reference_time falls in a gap between pairs: return the next
    pair (for starts) or the preceding pair (for ends, prefer_preceding).
    Clamped to the valid index range.
    """
    for pair_index, pair in enumerate(pairs):
        if reference_time < pair.start:
            if prefer_preceding and pair_index > 0:
                return pair_index - 1
            return pair_index
        if reference_time <= pair.end:
            return pair_index
    return len(pairs) - 1


def _format_seconds(time: timedelta) -> str:
    return f"{time.total_seconds():.3f}s"
