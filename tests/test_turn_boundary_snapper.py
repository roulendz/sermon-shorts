"""Tests for pipeline/turn_boundary_snapper.py."""

from datetime import timedelta

import srt

from models.video_segment import VideoSegment
from pipeline.turn_boundary_snapper import (
    build_speaking_turns,
    build_translation_pairs,
    extract_language_tag,
    snap_segments_to_translation_pairs,
    snap_time_window_to_pairs,
)


def make_subtitle(index, start_seconds, end_seconds, content):
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start_seconds),
        end=timedelta(seconds=end_seconds),
        content=content,
    )


def make_alternating_subtitles(pair_count, pair_duration_seconds=20.0):
    """Build [RU]+[LV] cue pairs: RU covers first 60%, LV the remaining 40%."""
    subtitles = []
    for pair_index in range(pair_count):
        pair_start = pair_index * pair_duration_seconds
        russian_end = pair_start + pair_duration_seconds * 0.6
        pair_end = pair_start + pair_duration_seconds
        subtitles.append(make_subtitle(
            2 * pair_index + 1, pair_start, russian_end, f"[RU] Оригинал {pair_index}",
        ))
        subtitles.append(make_subtitle(
            2 * pair_index + 2, russian_end, pair_end, f"[LV] Tulkojums {pair_index}",
        ))
    return subtitles


def make_segment(start_seconds, end_seconds):
    return VideoSegment(
        index=1,
        start_time=timedelta(seconds=start_seconds),
        end_time=timedelta(seconds=end_seconds),
        transcript_text="text",
        selection_reason="reason",
    )


# ── extract_language_tag ─────────────────────────────────────────────────────

def test_extract_language_tag_reads_russian_prefix():
    assert extract_language_tag("[RU] Привет") == "RU"


def test_extract_language_tag_reads_latvian_prefix():
    assert extract_language_tag("[LV] Sveiki") == "LV"


def test_extract_language_tag_returns_none_without_prefix():
    assert extract_language_tag("Plain text") is None


# ── build_speaking_turns ─────────────────────────────────────────────────────

def test_build_speaking_turns_groups_consecutive_same_language_cues():
    subtitles = [
        make_subtitle(1, 0, 5, "[RU] Раз"),
        make_subtitle(2, 5, 9, "[RU] Два"),
        make_subtitle(3, 9, 12, "[LV] Viens divi"),
        make_subtitle(4, 12, 18, "[RU] Три"),
    ]
    turns = build_speaking_turns(subtitles)
    assert [turn.language_tag for turn in turns] == ["RU", "LV", "RU"]
    assert turns[0].start == timedelta(seconds=0)
    assert turns[0].end == timedelta(seconds=9)
    assert turns[1].start == timedelta(seconds=9)


def test_build_speaking_turns_returns_empty_for_untagged_subtitles():
    subtitles = [make_subtitle(1, 0, 5, "No tag here")]
    assert build_speaking_turns(subtitles) == []


# ── build_translation_pairs ──────────────────────────────────────────────────

def test_build_translation_pairs_pairs_russian_with_following_latvian():
    turns = build_speaking_turns(make_alternating_subtitles(3))
    pairs = build_translation_pairs(turns)
    assert len(pairs) == 3
    for pair in pairs:
        assert pair.original_turn.language_tag == "RU"
        assert pair.translation_turn.language_tag == "LV"
        assert pair.start == pair.original_turn.start
        assert pair.end == pair.translation_turn.end


def test_build_translation_pairs_handles_leading_latvian_turn():
    subtitles = [
        make_subtitle(1, 0, 3, "[LV] Ievads"),
        make_subtitle(2, 3, 8, "[RU] Оригинал"),
        make_subtitle(3, 8, 11, "[LV] Tulkojums"),
    ]
    pairs = build_translation_pairs(build_speaking_turns(subtitles))
    assert len(pairs) == 2
    assert pairs[0].original_turn is None
    assert pairs[0].start == timedelta(seconds=0)


def test_build_translation_pairs_handles_trailing_untranslated_russian():
    subtitles = [
        make_subtitle(1, 0, 5, "[RU] Оригинал"),
        make_subtitle(2, 5, 8, "[LV] Tulkojums"),
        make_subtitle(3, 8, 12, "[RU] Без перевода"),
    ]
    pairs = build_translation_pairs(build_speaking_turns(subtitles))
    assert len(pairs) == 2
    assert pairs[1].translation_turn is None
    assert pairs[1].end == timedelta(seconds=12)


# ── snap_time_window_to_pairs ────────────────────────────────────────────────

def test_snap_moves_start_back_to_russian_turn_start():
    # 20s pairs: pair 3 spans 60-80 (RU 60-72, LV 72-80). Start lands in LV part.
    pairs = build_translation_pairs(
        build_speaking_turns(make_alternating_subtitles(10)),
    )
    snapped_start, _ = snap_time_window_to_pairs(
        timedelta(seconds=74), timedelta(seconds=130), pairs,
    )
    # Snapped to pair 3 start (60s) minus padding, clamped to pair 2 end (60s)
    assert snapped_start == timedelta(seconds=60)


def test_snap_moves_end_forward_through_latvian_translation():
    # End lands inside pair 6's RU turn (120-132); must extend to LV end 140.
    pairs = build_translation_pairs(
        build_speaking_turns(make_alternating_subtitles(10)),
    )
    _, snapped_end = snap_time_window_to_pairs(
        timedelta(seconds=62), timedelta(seconds=125), pairs,
    )
    # Pair 6 ends at 140s; padding clamped to next pair start (140s)
    assert snapped_end == timedelta(seconds=140)


def test_snap_drops_trailing_pairs_when_over_maximum_duration():
    pairs = build_translation_pairs(
        build_speaking_turns(make_alternating_subtitles(10)),
    )
    snapped_start, snapped_end = snap_time_window_to_pairs(
        timedelta(seconds=0), timedelta(seconds=199), pairs,
    )
    duration = (snapped_end - snapped_start).total_seconds()
    assert duration <= 105
    # End still lands exactly on a pair boundary (multiple of 20s)
    assert snapped_end.total_seconds() % 20 == 0


def test_snap_extends_forward_when_below_minimum_duration():
    pairs = build_translation_pairs(
        build_speaking_turns(make_alternating_subtitles(10)),
    )
    snapped_start, snapped_end = snap_time_window_to_pairs(
        timedelta(seconds=41), timedelta(seconds=55), pairs,
    )
    duration = (snapped_end - snapped_start).total_seconds()
    assert duration >= 45


def test_snap_keeps_single_oversized_pair_intact():
    # One giant pair longer than the maximum — cannot split, keep whole.
    subtitles = [
        make_subtitle(1, 0, 100, "[RU] Ļoti garš"),
        make_subtitle(2, 100, 120, "[LV] Ļoti garš tulkojums"),
    ]
    pairs = build_translation_pairs(build_speaking_turns(subtitles))
    snapped_start, snapped_end = snap_time_window_to_pairs(
        timedelta(seconds=10), timedelta(seconds=110), pairs,
    )
    assert snapped_start == timedelta(seconds=0)
    assert snapped_end >= timedelta(seconds=120)


def test_snap_start_padding_applies_before_first_pair():
    pairs = build_translation_pairs(
        build_speaking_turns(make_alternating_subtitles(5)),
    )
    snapped_start, _ = snap_time_window_to_pairs(
        timedelta(seconds=1), timedelta(seconds=55), pairs,
    )
    assert snapped_start == timedelta(seconds=0)


def test_snap_end_padding_extends_into_gap_after_last_pair():
    subtitles = [
        make_subtitle(1, 0, 30, "[RU] Оригинал"),
        make_subtitle(2, 30, 50, "[LV] Tulkojums"),
    ]
    pairs = build_translation_pairs(build_speaking_turns(subtitles))
    _, snapped_end = snap_time_window_to_pairs(
        timedelta(seconds=5), timedelta(seconds=45), pairs,
    )
    assert snapped_end == timedelta(seconds=50, milliseconds=300)


# ── snap_segments_to_translation_pairs ───────────────────────────────────────

def test_snap_segments_mutates_boundaries_in_place():
    subtitles = make_alternating_subtitles(10)
    segment = make_segment(74, 125)
    snap_segments_to_translation_pairs([segment], subtitles)
    assert segment.start_time == timedelta(seconds=60)
    assert segment.end_time == timedelta(seconds=140)


def test_snap_segments_is_noop_for_untagged_subtitles():
    subtitles = [
        make_subtitle(1, 0, 60, "Vienvalodas teksts"),
        make_subtitle(2, 60, 130, "Vēl teksts"),
    ]
    segment = make_segment(10, 70)
    snap_segments_to_translation_pairs([segment], subtitles)
    assert segment.start_time == timedelta(seconds=10)
    assert segment.end_time == timedelta(seconds=70)
