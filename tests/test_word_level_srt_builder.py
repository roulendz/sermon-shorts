"""Tests for pipeline/word_level_srt_builder.py"""

from datetime import timedelta
from pathlib import Path

from pipeline.word_level_srt_builder import (
    build_word_level_subtitles,
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


def test_merge_bilingual_word_level_srt_interleaves_by_timestamp():
    ru_response = {
        "segments": [
            {"start": 10.0, "end": 15.0, "text": "Privet mir"},
            {"start": 25.0, "end": 30.0, "text": "Eto test"},
        ]
    }
    lv_response = {
        "segments": [
            {"start": 16.0, "end": 22.0, "text": "Sveika pasaule"},
            {"start": 31.0, "end": 36.0, "text": "Tas ir tests"},
        ]
    }

    ru_subtitles = build_word_level_subtitles(ru_response)
    lv_subtitles = build_word_level_subtitles(lv_response)

    merged = merge_bilingual_word_level_srt(
        primary_subtitles=ru_subtitles,
        primary_language_tag="RU",
        secondary_subtitles=lv_subtitles,
        secondary_language_tag="LV",
    )

    assert len(merged) == 4
    # Should be sorted by time: RU(10), LV(16), RU(25), LV(31)
    assert merged[0].content == "[RU] Privet mir"
    assert merged[1].content == "[LV] Sveika pasaule"
    assert merged[2].content == "[RU] Eto test"
    assert merged[3].content == "[LV] Tas ir tests"
    # Indexes should be sequential
    assert [subtitle.index for subtitle in merged] == [1, 2, 3, 4]


def test_merge_bilingual_word_level_srt_handles_single_language():
    response = {
        "segments": [
            {"start": 5.0, "end": 10.0, "text": "Only one language"},
        ]
    }
    subtitles = build_word_level_subtitles(response)

    merged = merge_bilingual_word_level_srt(
        primary_subtitles=subtitles,
        primary_language_tag="LV",
        secondary_subtitles=[],
        secondary_language_tag="RU",
    )

    assert len(merged) == 1
    assert merged[0].content == "[LV] Only one language"


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
