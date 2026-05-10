"""
tests/test_subtitle_parser.py
Tests for pipeline/subtitle_parser.py
"""

import pytest
import srt
from datetime import timedelta
from pathlib import Path

from pipeline.subtitle_parser import (
    find_end_of_preceding_subtitle,
    parse_subtitle_text,
    shift_subtitle_timestamps,
    slice_subtitles_within_window,
    reindex_and_rebase_subtitles,
    extract_plain_text_from_subtitles,
    serialize_subtitles_to_srt_text,
    save_subtitles_to_file,
)


SAMPLE_SRT_TEXT = """\
1
00:01:00,000 --> 00:01:05,000
Hello world.

2
00:01:06,000 --> 00:01:10,000
This is a sermon.

3
00:05:00,000 --> 00:05:08,000
God is good.

4
00:05:09,000 --> 00:05:20,000
All the time.

5
00:10:00,000 --> 00:10:15,000
Final words here.
"""


def test_parse_subtitle_text_returns_correct_number_of_entries():
    subtitles = parse_subtitle_text(SAMPLE_SRT_TEXT)
    assert len(subtitles) == 5


def test_parse_subtitle_text_preserves_content():
    subtitles = parse_subtitle_text(SAMPLE_SRT_TEXT)
    assert subtitles[0].content == "Hello world."
    assert subtitles[4].content == "Final words here."


def test_parse_subtitle_text_preserves_timestamps():
    subtitles = parse_subtitle_text(SAMPLE_SRT_TEXT)
    assert subtitles[0].start == timedelta(minutes=1)
    assert subtitles[0].end == timedelta(minutes=1, seconds=5)


def test_shift_subtitle_timestamps_shifts_all_entries_by_offset(simple_subtitle_list):
    offset = timedelta(seconds=45)
    shifted = shift_subtitle_timestamps(simple_subtitle_list, offset)

    assert shifted[0].start == simple_subtitle_list[0].start + offset
    assert shifted[0].end == simple_subtitle_list[0].end + offset


def test_shift_subtitle_timestamps_shifts_every_entry(simple_subtitle_list):
    offset = timedelta(seconds=10)
    shifted = shift_subtitle_timestamps(simple_subtitle_list, offset)
    assert len(shifted) == len(simple_subtitle_list)


def test_shift_subtitle_timestamps_drops_entries_that_become_negative():
    subtitles = [
        srt.Subtitle(index=1, start=timedelta(seconds=2), end=timedelta(seconds=5), content="Early."),
        srt.Subtitle(index=2, start=timedelta(seconds=10), end=timedelta(seconds=15), content="Later."),
    ]
    # Negative offset shifts backwards — first entry goes negative
    shifted = shift_subtitle_timestamps(subtitles, timedelta(seconds=-5))
    assert len(shifted) == 1
    assert shifted[0].content == "Later."


def test_shift_subtitle_timestamps_does_not_mutate_original(simple_subtitle_list):
    original_start = simple_subtitle_list[0].start
    shift_subtitle_timestamps(simple_subtitle_list, timedelta(seconds=100))
    assert simple_subtitle_list[0].start == original_start


def test_slice_subtitles_within_window_returns_only_entries_in_range(simple_subtitle_list):
    window_start = timedelta(minutes=5)
    window_end = timedelta(minutes=5, seconds=25)
    sliced = slice_subtitles_within_window(simple_subtitle_list, window_start, window_end)

    assert len(sliced) == 2
    assert sliced[0].content == "God is good."
    assert sliced[1].content == "All the time."


def test_slice_subtitles_within_window_reindexes_from_one(simple_subtitle_list):
    window_start = timedelta(minutes=5)
    window_end = timedelta(minutes=5, seconds=25)
    sliced = slice_subtitles_within_window(simple_subtitle_list, window_start, window_end)

    assert sliced[0].index == 1
    assert sliced[1].index == 2


def test_slice_subtitles_within_window_rebases_timestamps_to_zero(simple_subtitle_list):
    window_start = timedelta(minutes=5)
    window_end = timedelta(minutes=5, seconds=25)
    sliced = slice_subtitles_within_window(simple_subtitle_list, window_start, window_end)

    # Original start was 5:00, window_start is 5:00, so rebased start should be 0:00
    assert sliced[0].start == timedelta(0)


def test_slice_subtitles_within_window_returns_empty_when_no_entries_in_range(simple_subtitle_list):
    sliced = slice_subtitles_within_window(
        simple_subtitle_list,
        window_start=timedelta(hours=2),
        window_end=timedelta(hours=3),
    )
    assert sliced == []


def test_reindex_and_rebase_subtitles_starts_index_at_one(simple_subtitle_list):
    rebased = reindex_and_rebase_subtitles(simple_subtitle_list, timedelta(minutes=1))
    assert rebased[0].index == 1
    assert rebased[1].index == 2


def test_reindex_and_rebase_subtitles_subtracts_time_base(simple_subtitle_list):
    time_base = timedelta(minutes=1)
    rebased = reindex_and_rebase_subtitles(simple_subtitle_list, time_base)
    assert rebased[0].start == simple_subtitle_list[0].start - time_base


def test_extract_plain_text_joins_all_content(simple_subtitle_list):
    text = extract_plain_text_from_subtitles(simple_subtitle_list)
    assert "Hello world." in text
    assert "God is good." in text
    assert "Final words here." in text


def test_extract_plain_text_strips_whitespace_from_entries():
    subtitles = [
        srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="  Hello  "),
        srt.Subtitle(index=2, start=timedelta(seconds=3), end=timedelta(seconds=4), content="  World  "),
    ]
    text = extract_plain_text_from_subtitles(subtitles)
    assert text == "Hello World"


def test_serialize_subtitles_to_srt_text_produces_parseable_output(simple_subtitle_list):
    srt_text = serialize_subtitles_to_srt_text(simple_subtitle_list)
    reparsed = parse_subtitle_text(srt_text)
    assert len(reparsed) == len(simple_subtitle_list)
    assert reparsed[0].content == simple_subtitle_list[0].content


def test_save_subtitles_to_file_writes_readable_srt(simple_subtitle_list, tmp_path):
    output_path = tmp_path / "test_output.srt"
    save_subtitles_to_file(simple_subtitle_list, output_path)

    assert output_path.exists()
    written_text = output_path.read_text(encoding="utf-8")
    reparsed = parse_subtitle_text(written_text)
    assert len(reparsed) == len(simple_subtitle_list)


def test_save_subtitles_to_file_creates_parent_directories(simple_subtitle_list, tmp_path):
    nested_path = tmp_path / "deeply" / "nested" / "output.srt"
    save_subtitles_to_file(simple_subtitle_list, nested_path)
    assert nested_path.exists()


def test_find_end_of_preceding_subtitle_returns_previous_end(simple_subtitle_list):
    # reference_time in gap between sub 2 (end 1:10) and sub 3 (start 5:00)
    result = find_end_of_preceding_subtitle(simple_subtitle_list, timedelta(minutes=2))
    assert result == timedelta(minutes=1, seconds=10)


def test_find_end_of_preceding_subtitle_at_exact_start(simple_subtitle_list):
    # reference_time exactly at sub 3 start (5:00) — should return sub 2 end
    result = find_end_of_preceding_subtitle(simple_subtitle_list, timedelta(minutes=5))
    assert result == timedelta(minutes=1, seconds=10)


def test_find_end_of_preceding_subtitle_returns_none_for_first(simple_subtitle_list):
    # reference_time before or at first subtitle — no preceding subtitle
    result = find_end_of_preceding_subtitle(simple_subtitle_list, timedelta(seconds=30))
    assert result is None


def test_find_end_of_preceding_subtitle_returns_none_for_empty_list():
    result = find_end_of_preceding_subtitle([], timedelta(minutes=1))
    assert result is None


def test_find_end_of_preceding_subtitle_returns_none_when_all_after():
    subtitles = [
        srt.Subtitle(index=1, start=timedelta(minutes=10), end=timedelta(minutes=10, seconds=5), content="Late."),
    ]
    result = find_end_of_preceding_subtitle(subtitles, timedelta(minutes=10))
    assert result is None


def test_find_end_of_preceding_subtitle_consecutive_subtitles():
    # Two consecutive subs with no gap — preceding end equals current start
    subtitles = [
        srt.Subtitle(index=1, start=timedelta(seconds=10), end=timedelta(seconds=20), content="A"),
        srt.Subtitle(index=2, start=timedelta(seconds=20), end=timedelta(seconds=30), content="B"),
        srt.Subtitle(index=3, start=timedelta(seconds=40), end=timedelta(seconds=50), content="C"),
    ]
    result = find_end_of_preceding_subtitle(subtitles, timedelta(seconds=20))
    assert result == timedelta(seconds=20)


def test_find_end_of_preceding_subtitle_reference_inside_subtitle():
    """When reference_time falls inside a long subtitle, return that subtitle's
    end — the boundary where untranscribed (Russian) audio begins before
    the next Latvian subtitle."""
    subtitles = [
        srt.Subtitle(index=64, start=timedelta(minutes=10, seconds=52, milliseconds=261),
                      end=timedelta(minutes=10, seconds=53, milliseconds=761), content="Grib."),
        srt.Subtitle(index=65, start=timedelta(minutes=10, seconds=56, milliseconds=301),
                      end=timedelta(minutes=11, seconds=35, milliseconds=482), content="Long Latvian passage."),
        srt.Subtitle(index=66, start=timedelta(minutes=11, seconds=37, milliseconds=202),
                      end=timedelta(minutes=11, seconds=38, milliseconds=42), content="Jūs sakot man?"),
    ]
    reference_time = timedelta(minutes=11, seconds=20, milliseconds=872)
    result = find_end_of_preceding_subtitle(subtitles, reference_time)
    assert result == timedelta(minutes=11, seconds=35, milliseconds=482)
