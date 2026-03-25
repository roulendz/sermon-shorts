"""Tests for pipeline/word_level_srt_builder.py"""

from datetime import timedelta
from pathlib import Path

from pipeline.word_level_srt_builder import (
    build_word_level_subtitles,
    extract_words_from_response,
    map_words_to_speech_regions,
    merge_bilingual_word_level_srt,
    save_word_level_srt,
)


SAMPLE_WHISPERX_RESPONSE_WITH_WORDS = {
    "segments": [
        {
            "start": 10.0,
            "end": 15.0,
            "text": "Hello world today",
            "words": [
                {"word": "Hello", "start": 10.2, "end": 10.8, "score": 0.95},
                {"word": "world", "start": 11.0, "end": 11.5, "score": 0.92},
                {"word": "today", "start": 11.8, "end": 12.3, "score": 0.88},
            ],
        },
        {
            "start": 16.0,
            "end": 20.0,
            "text": "This is a test",
            "words": [
                {"word": "This", "start": 16.1, "end": 16.4, "score": 0.90},
                {"word": "is", "start": 16.5, "end": 16.7, "score": 0.85},
                {"word": "a", "start": 16.8, "end": 16.9, "score": 0.80},
                {"word": "test", "start": 17.0, "end": 17.5, "score": 0.93},
            ],
        },
    ]
}

SAMPLE_WHISPERX_RESPONSE_WITHOUT_WORDS = {
    "segments": [
        {"start": 5.0, "end": 10.0, "text": "No word data here"},
        {"start": 11.0, "end": 15.0, "text": "Still no words"},
    ]
}


def test_build_word_level_subtitles_uses_word_timestamps():
    subtitles = build_word_level_subtitles(SAMPLE_WHISPERX_RESPONSE_WITH_WORDS)

    assert len(subtitles) == 2

    # First subtitle should use word-level start (10.2) and end (12.3)
    assert subtitles[0].start == timedelta(seconds=10.2)
    assert subtitles[0].end == timedelta(seconds=12.3)
    assert subtitles[0].content == "Hello world today"

    # Second subtitle should use word-level start (16.1) and end (17.5)
    assert subtitles[1].start == timedelta(seconds=16.1)
    assert subtitles[1].end == timedelta(seconds=17.5)
    assert subtitles[1].content == "This is a test"


def test_build_word_level_subtitles_falls_back_to_segment_timestamps():
    subtitles = build_word_level_subtitles(SAMPLE_WHISPERX_RESPONSE_WITHOUT_WORDS)

    assert len(subtitles) == 2

    # Should use segment-level timestamps
    assert subtitles[0].start == timedelta(seconds=5.0)
    assert subtitles[0].end == timedelta(seconds=10.0)

    assert subtitles[1].start == timedelta(seconds=11.0)
    assert subtitles[1].end == timedelta(seconds=15.0)


def test_build_word_level_subtitles_handles_nested_result():
    response = {"result": SAMPLE_WHISPERX_RESPONSE_WITH_WORDS}

    subtitles = build_word_level_subtitles(response)

    assert len(subtitles) == 2


def test_build_word_level_subtitles_skips_empty_text():
    response = {
        "segments": [
            {"start": 1.0, "end": 2.0, "text": ""},
            {"start": 3.0, "end": 4.0, "text": "  "},
            {"start": 5.0, "end": 6.0, "text": "Real content"},
        ]
    }

    subtitles = build_word_level_subtitles(response)

    assert len(subtitles) == 1
    assert subtitles[0].content == "Real content"


def test_build_word_level_subtitles_skips_zero_duration():
    response = {
        "segments": [
            {"start": 5.0, "end": 5.0, "text": "Zero duration"},
            {"start": 6.0, "end": 10.0, "text": "Valid segment"},
        ]
    }

    subtitles = build_word_level_subtitles(response)

    assert len(subtitles) == 1
    assert subtitles[0].content == "Valid segment"


def test_build_word_level_subtitles_handles_missing_word_start():
    response = {
        "segments": [
            {
                "start": 5.0,
                "end": 10.0,
                "text": "Partial word data",
                "words": [
                    {"word": "Partial", "end": 6.0, "score": 0.9},
                    {"word": "word", "start": 7.0, "end": 8.0, "score": 0.9},
                    {"word": "data", "start": 8.5, "end": 9.0, "score": 0.9},
                ],
            }
        ]
    }

    subtitles = build_word_level_subtitles(response)

    assert len(subtitles) == 1
    # First word has no start, should use second word's start (7.0)
    assert subtitles[0].start == timedelta(seconds=7.0)
    assert subtitles[0].end == timedelta(seconds=9.0)


def test_build_word_level_subtitles_handles_empty_response():
    subtitles = build_word_level_subtitles({"segments": []})

    assert subtitles == []


def test_build_word_level_subtitles_handles_list_response():
    response = [
        {"start": 1.0, "end": 3.0, "text": "Direct list format"},
    ]

    subtitles = build_word_level_subtitles(response)

    assert len(subtitles) == 1


def test_map_words_to_speech_regions_basic():
    words = [
        {"word": "Privet", "start": 10.2, "end": 10.8},
        {"word": "mir", "start": 11.0, "end": 11.5},
        {"word": "Kak", "start": 25.1, "end": 25.5},
        {"word": "dela", "start": 25.6, "end": 26.0},
    ]
    speech_regions = [(10.0, 15.0), (25.0, 30.0)]

    subtitles = map_words_to_speech_regions(words, speech_regions, "RU")

    assert len(subtitles) == 2
    assert subtitles[0].content == "[RU] Privet mir"
    assert subtitles[0].start == timedelta(seconds=10.2)
    assert subtitles[0].end == timedelta(seconds=11.5)
    assert subtitles[1].content == "[RU] Kak dela"


def test_map_words_to_speech_regions_skips_empty_regions():
    words = [
        {"word": "Hello", "start": 10.5, "end": 11.0},
    ]
    speech_regions = [(5.0, 8.0), (10.0, 15.0)]

    subtitles = map_words_to_speech_regions(words, speech_regions, "LV")

    assert len(subtitles) == 1
    assert subtitles[0].content == "[LV] Hello"


def test_merge_bilingual_word_level_srt_interleaves_by_timestamp():
    ru_words = [
        {"word": "Privet", "start": 10.2, "end": 10.8},
        {"word": "Eto", "start": 25.1, "end": 25.5},
    ]
    lv_words = [
        {"word": "Sveiki", "start": 16.1, "end": 16.8},
        {"word": "Tests", "start": 31.1, "end": 31.5},
    ]
    ru_regions = [(10.0, 15.0), (25.0, 30.0)]
    lv_regions = [(16.0, 22.0), (31.0, 36.0)]

    ru_subs = map_words_to_speech_regions(ru_words, ru_regions, "RU")
    lv_subs = map_words_to_speech_regions(lv_words, lv_regions, "LV")

    merged = merge_bilingual_word_level_srt(ru_subs, lv_subs)

    assert len(merged) == 4
    assert merged[0].content == "[RU] Privet"
    assert merged[1].content == "[LV] Sveiki"
    assert merged[2].content == "[RU] Eto"
    assert merged[3].content == "[LV] Tests"
    assert [subtitle.index for subtitle in merged] == [1, 2, 3, 4]


def test_merge_bilingual_word_level_srt_handles_single_language():
    words = [{"word": "Only", "start": 5.5, "end": 6.0}]
    regions = [(5.0, 10.0)]

    subs = map_words_to_speech_regions(words, regions, "LV")
    merged = merge_bilingual_word_level_srt(subs, [])

    assert len(merged) == 1
    assert merged[0].content == "[LV] Only"


def test_extract_words_from_response():
    response = {
        "segments": [
            {
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 1.0, "end": 1.5},
                    {"word": "world", "start": 1.6, "end": 2.0},
                ],
            },
            {
                "text": "No timestamps",
                "words": [
                    {"word": "No"},
                    {"word": "timestamps"},
                ],
            },
        ]
    }

    words = extract_words_from_response(response)

    assert len(words) == 2
    assert words[0]["word"] == "Hello"
    assert words[1]["word"] == "world"


def test_save_word_level_srt_creates_file(tmp_path):
    subtitles = build_word_level_subtitles(SAMPLE_WHISPERX_RESPONSE_WITH_WORDS)

    output_path = tmp_path / "output" / "test.srt"
    result = save_word_level_srt(subtitles, output_path)

    assert result == output_path
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")
    assert "Hello world today" in content
    assert "This is a test" in content
    assert "-->" in content
