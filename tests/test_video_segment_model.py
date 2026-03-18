"""
tests/test_video_segment_model.py
Tests for models/video_segment.py
"""

import pytest
from datetime import timedelta
from models.video_segment import VideoSegment


def make_valid_segment(**overrides) -> VideoSegment:
    defaults = dict(
        index=1,
        start_time=timedelta(minutes=5),
        end_time=timedelta(minutes=5, seconds=25),
        transcript_text="God is good all the time.",
        selection_reason="Strong hook",
    )
    defaults.update(overrides)
    return VideoSegment(**defaults)


def test_duration_is_difference_between_end_and_start_time():
    segment = make_valid_segment(
        start_time=timedelta(minutes=5),
        end_time=timedelta(minutes=5, seconds=25),
    )
    assert segment.duration == timedelta(seconds=25)


def test_duration_seconds_returns_float():
    segment = make_valid_segment(
        start_time=timedelta(minutes=1),
        end_time=timedelta(minutes=1, seconds=30),
    )
    assert segment.duration_seconds == 30.0


def test_start_seconds_returns_total_seconds_of_start_time():
    segment = make_valid_segment(start_time=timedelta(minutes=2, seconds=15))
    assert segment.start_seconds == 135.0


def test_end_seconds_returns_total_seconds_of_end_time():
    segment = make_valid_segment(
        start_time=timedelta(minutes=2),
        end_time=timedelta(minutes=3),
    )
    assert segment.end_seconds == 180.0


def test_ffmpeg_start_timestamp_formats_correctly():
    segment = make_valid_segment(
        start_time=timedelta(hours=1, minutes=23, seconds=45, milliseconds=678),
        end_time=timedelta(hours=1, minutes=24, seconds=10),
    )
    assert segment.ffmpeg_start_timestamp == "01:23:45.678"


def test_ffmpeg_end_timestamp_formats_correctly():
    segment = make_valid_segment(end_time=timedelta(hours=0, minutes=5, seconds=9, milliseconds=100))
    assert segment.ffmpeg_end_timestamp == "00:05:09.100"


def test_ffmpeg_timestamp_zero_pads_hours_minutes_seconds():
    segment = make_valid_segment(start_time=timedelta(seconds=3, milliseconds=5))
    assert segment.ffmpeg_start_timestamp == "00:00:03.005"


def test_raises_value_error_when_end_time_is_before_start_time():
    with pytest.raises(ValueError, match="end_time"):
        make_valid_segment(
            start_time=timedelta(minutes=10),
            end_time=timedelta(minutes=5),
        )


def test_raises_value_error_when_end_time_equals_start_time():
    with pytest.raises(ValueError, match="end_time"):
        make_valid_segment(
            start_time=timedelta(minutes=5),
            end_time=timedelta(minutes=5),
        )


def test_raises_value_error_when_start_time_is_negative():
    with pytest.raises(ValueError, match="start_time"):
        make_valid_segment(start_time=timedelta(seconds=-1))


def test_raises_value_error_when_transcript_text_is_empty():
    with pytest.raises(ValueError, match="transcript_text"):
        make_valid_segment(transcript_text="")


def test_raises_value_error_when_transcript_text_is_only_whitespace():
    with pytest.raises(ValueError, match="transcript_text"):
        make_valid_segment(transcript_text="   ")


def test_repr_contains_index_and_timestamps():
    segment = make_valid_segment(index=3)
    representation = repr(segment)
    assert "index=3" in representation
    assert "00:05:00" in representation


# ── Scoring fields ────────────────────────────────────────────────────────────

def test_scoring_fields_default_to_zero():
    segment = make_valid_segment()
    assert segment.hook_power_score == 0
    assert segment.emotional_impact_score == 0
    assert segment.discussion_potential_score == 0
    assert segment.shareability_score == 0
    assert segment.overall_viral_score == 0.0


def test_viral_hook_text_overlay_defaults_to_empty_string():
    segment = make_valid_segment()
    assert segment.viral_hook_text_overlay == ""


def test_scoring_fields_accept_provided_values():
    segment = make_valid_segment(
        hook_power_score=8,
        emotional_impact_score=7,
        discussion_potential_score=6,
        shareability_score=9,
        overall_viral_score=7.5,
        viral_hook_text_overlay="You need to hear this",
    )
    assert segment.hook_power_score == 8
    assert segment.emotional_impact_score == 7
    assert segment.discussion_potential_score == 6
    assert segment.shareability_score == 9
    assert segment.overall_viral_score == 7.5
    assert segment.viral_hook_text_overlay == "You need to hear this"


def test_repr_includes_viral_score():
    segment = make_valid_segment(overall_viral_score=8.3)
    assert "8.3" in repr(segment)
