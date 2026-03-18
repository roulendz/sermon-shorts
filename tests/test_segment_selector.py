"""
tests/test_segment_selector.py
Tests for pipeline/segment_selector.py
"""

import pytest
import srt
from datetime import timedelta

from pipeline.segment_selector import (
    build_segment_selection_prompt,
    parse_segments_from_manus_response,
    extract_json_from_response_text,
    convert_seconds_float_to_timedelta,
    extract_transcript_text_for_window,
    MINIMUM_CLIPS,
    MAXIMUM_CLIPS,
)


VALID_MANUS_RESPONSE_JSON = """{
  "potential_clips": [
    {
      "index": 1,
      "start_time": 300.0,
      "end_time": 320.0,
      "suggested_title": "God Is Always On Time",
      "viral_hook_text_overlay": "You need to hear this truth",
      "scoring_analysis": {
        "hook_power_score": 8,
        "emotional_impact_score": 7,
        "discussion_potential_score": 6,
        "shareability_score": 9,
        "overall_viral_score": 7.5
      },
      "reason_for_selection": "Strong emotional hook with a relatable message"
    },
    {
      "index": 2,
      "start_time": 600.0,
      "end_time": 615.0,
      "suggested_title": "This Changed Everything",
      "viral_hook_text_overlay": "Nobody talks about this",
      "scoring_analysis": {
        "hook_power_score": 9,
        "emotional_impact_score": 8,
        "discussion_potential_score": 7,
        "shareability_score": 8,
        "overall_viral_score": 8.3
      },
      "reason_for_selection": "Clear call to action with high shareability"
    }
  ]
}"""

VALID_MANUS_RESPONSE_WITH_MARKDOWN_FENCES = f"```json\n{VALID_MANUS_RESPONSE_JSON}\n```"


def make_subtitle_list_for_window_test() -> list[srt.Subtitle]:
    return [
        srt.Subtitle(index=1, start=timedelta(minutes=5), end=timedelta(minutes=5, seconds=8), content="God is good."),
        srt.Subtitle(index=2, start=timedelta(minutes=5, seconds=9), end=timedelta(minutes=5, seconds=20), content="All the time."),
        srt.Subtitle(index=3, start=timedelta(minutes=10), end=timedelta(minutes=10, seconds=15), content="Final blessing."),
    ]


# ── Prompt builder ────────────────────────────────────────────────────────────

def test_build_segment_selection_prompt_includes_srt_content():
    srt_content = "1\n00:01:00,000 --> 00:01:05,000\nHello world.\n"
    prompt = build_segment_selection_prompt(srt_content)
    assert srt_content in prompt


def test_build_segment_selection_prompt_includes_minimum_clip_count():
    prompt = build_segment_selection_prompt("some srt content")
    assert str(MINIMUM_CLIPS) in prompt


def test_build_segment_selection_prompt_includes_maximum_clip_count():
    prompt = build_segment_selection_prompt("some srt content")
    assert str(MAXIMUM_CLIPS) in prompt


def test_build_segment_selection_prompt_requests_json_only_response():
    prompt = build_segment_selection_prompt("some srt content")
    assert "JSON" in prompt


def test_build_segment_selection_prompt_mentions_tiktok_and_instagram():
    prompt = build_segment_selection_prompt("some srt content")
    assert "TikTok" in prompt
    assert "Instagram" in prompt


def test_build_segment_selection_prompt_requests_absolute_seconds_timestamps():
    prompt = build_segment_selection_prompt("some srt content")
    assert "ABSOLUTE SECONDS" in prompt


def test_build_segment_selection_prompt_does_not_contain_video_duration_placeholder():
    prompt = build_segment_selection_prompt("some srt content")
    assert "{video_duration" not in prompt


def test_build_segment_selection_prompt_includes_all_four_scoring_dimensions():
    prompt = build_segment_selection_prompt("some srt content")
    assert "hook_power" in prompt
    assert "emotional_impact" in prompt
    assert "discussion_potential" in prompt
    assert "shareability" in prompt


# ── JSON extractor ────────────────────────────────────────────────────────────

def test_extract_json_from_response_text_parses_clean_json():
    parsed = extract_json_from_response_text(VALID_MANUS_RESPONSE_JSON)
    assert "potential_clips" in parsed


def test_extract_json_from_response_text_strips_markdown_fences():
    parsed = extract_json_from_response_text(VALID_MANUS_RESPONSE_WITH_MARKDOWN_FENCES)
    assert "potential_clips" in parsed


def test_extract_json_from_response_text_raises_on_invalid_json():
    with pytest.raises(ValueError, match="Could not parse JSON"):
        extract_json_from_response_text("this is not json at all")


def test_extract_json_from_response_text_handles_surrounding_whitespace():
    padded = f"\n\n  {VALID_MANUS_RESPONSE_JSON}  \n\n"
    parsed = extract_json_from_response_text(padded)
    assert "potential_clips" in parsed


# ── Float seconds converter ───────────────────────────────────────────────────

def test_convert_seconds_float_to_timedelta_converts_whole_seconds():
    result = convert_seconds_float_to_timedelta(60.0)
    assert result == timedelta(minutes=1)


def test_convert_seconds_float_to_timedelta_converts_fractional_seconds():
    result = convert_seconds_float_to_timedelta(95.350)
    assert abs(result.total_seconds() - 95.350) < 0.001


def test_convert_seconds_float_to_timedelta_converts_zero():
    result = convert_seconds_float_to_timedelta(0.0)
    assert result == timedelta(0)


def test_convert_seconds_float_to_timedelta_converts_large_value():
    result = convert_seconds_float_to_timedelta(3661.0)
    assert result == timedelta(hours=1, minutes=1, seconds=1)


# ── Transcript text extractor ─────────────────────────────────────────────────

def test_extract_transcript_text_for_window_returns_text_within_range():
    subtitles = make_subtitle_list_for_window_test()
    text = extract_transcript_text_for_window(
        subtitles,
        window_start=timedelta(minutes=5),
        window_end=timedelta(minutes=5, seconds=25),
    )
    assert "God is good." in text
    assert "All the time." in text


def test_extract_transcript_text_for_window_excludes_text_outside_range():
    subtitles = make_subtitle_list_for_window_test()
    text = extract_transcript_text_for_window(
        subtitles,
        window_start=timedelta(minutes=5),
        window_end=timedelta(minutes=5, seconds=25),
    )
    assert "Final blessing." not in text


def test_extract_transcript_text_for_window_returns_empty_string_when_no_match():
    subtitles = make_subtitle_list_for_window_test()
    text = extract_transcript_text_for_window(
        subtitles,
        window_start=timedelta(hours=2),
        window_end=timedelta(hours=3),
    )
    assert text == ""


# ── Full response parser ──────────────────────────────────────────────────────

def test_parse_segments_from_manus_response_returns_correct_segment_count():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert len(segments) == 2


def test_parse_segments_from_manus_response_converts_float_start_time_to_timedelta():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].start_time == timedelta(seconds=300.0)


def test_parse_segments_from_manus_response_converts_float_end_time_to_timedelta():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].end_time == timedelta(seconds=320.0)


def test_parse_segments_from_manus_response_populates_hook_power_score():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].hook_power_score == 8


def test_parse_segments_from_manus_response_populates_emotional_impact_score():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].emotional_impact_score == 7


def test_parse_segments_from_manus_response_populates_discussion_potential_score():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].discussion_potential_score == 6


def test_parse_segments_from_manus_response_populates_shareability_score():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].shareability_score == 9


def test_parse_segments_from_manus_response_populates_overall_viral_score():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].overall_viral_score == 7.5


def test_parse_segments_from_manus_response_populates_viral_hook_text_overlay():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].viral_hook_text_overlay == "You need to hear this truth"


def test_parse_segments_from_manus_response_populates_selection_reason():
    subtitles = make_subtitle_list_for_window_test()
    segments = parse_segments_from_manus_response(VALID_MANUS_RESPONSE_JSON, subtitles)
    assert segments[0].selection_reason == "Strong emotional hook with a relatable message"


def test_parse_segments_from_manus_response_raises_when_potential_clips_is_empty():
    empty_response = '{"potential_clips": []}'
    with pytest.raises(ValueError, match="no segments"):
        parse_segments_from_manus_response(empty_response, [])
