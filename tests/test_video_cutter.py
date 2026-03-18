"""
tests/test_video_cutter.py
Tests for pipeline/video_cutter.py

Tests the FFmpeg argument builder as a pure function.
Does not execute FFmpeg — only verifies the correct arguments are produced.
"""

import pytest
from pathlib import Path
from datetime import timedelta

from pipeline.video_cutter import (
    build_ffmpeg_cut_arguments,
    build_output_video_file_path,
    build_output_subtitle_file_path,
)
from models.video_segment import VideoSegment


def make_segment_at(
    start: timedelta,
    end: timedelta,
    index: int = 1,
) -> VideoSegment:
    return VideoSegment(
        index=index,
        start_time=start,
        end_time=end,
        transcript_text="Test transcript text.",
        selection_reason="Test reason",
    )


def test_build_ffmpeg_cut_arguments_starts_with_ffmpeg():
    args = build_ffmpeg_cut_arguments(
        source_video_path=Path("video.mp4"),
        start_timestamp="00:05:00.000",
        end_timestamp="00:05:25.000",
        output_file_path=Path("output/segment_01.mp4"),
    )
    assert args[0] == "ffmpeg"


def test_build_ffmpeg_cut_arguments_includes_overwrite_flag():
    args = build_ffmpeg_cut_arguments(
        source_video_path=Path("video.mp4"),
        start_timestamp="00:05:00.000",
        end_timestamp="00:05:25.000",
        output_file_path=Path("output/segment_01.mp4"),
    )
    assert "-y" in args


def test_build_ffmpeg_cut_arguments_places_start_timestamp_before_input():
    args = build_ffmpeg_cut_arguments(
        source_video_path=Path("video.mp4"),
        start_timestamp="00:05:00.000",
        end_timestamp="00:05:25.000",
        output_file_path=Path("output/segment_01.mp4"),
    )
    start_index = args.index("-ss")
    input_index = args.index("-i")
    assert start_index < input_index


def test_build_ffmpeg_cut_arguments_includes_start_timestamp():
    args = build_ffmpeg_cut_arguments(
        source_video_path=Path("video.mp4"),
        start_timestamp="00:05:00.000",
        end_timestamp="00:05:25.000",
        output_file_path=Path("output/segment_01.mp4"),
    )
    start_index = args.index("-ss")
    assert args[start_index + 1] == "00:05:00.000"


def test_build_ffmpeg_cut_arguments_includes_end_timestamp():
    args = build_ffmpeg_cut_arguments(
        source_video_path=Path("video.mp4"),
        start_timestamp="00:05:00.000",
        end_timestamp="00:05:25.000",
        output_file_path=Path("output/segment_01.mp4"),
    )
    end_index = args.index("-to")
    assert args[end_index + 1] == "00:05:25.000"


def test_build_ffmpeg_cut_arguments_uses_stream_copy_for_no_reencode():
    args = build_ffmpeg_cut_arguments(
        source_video_path=Path("video.mp4"),
        start_timestamp="00:05:00.000",
        end_timestamp="00:05:25.000",
        output_file_path=Path("output/segment_01.mp4"),
    )
    codec_index = args.index("-c")
    assert args[codec_index + 1] == "copy"


def test_build_ffmpeg_cut_arguments_ends_with_output_path():
    output_path = Path("output/segment_01.mp4")
    args = build_ffmpeg_cut_arguments(
        source_video_path=Path("video.mp4"),
        start_timestamp="00:05:00.000",
        end_timestamp="00:05:25.000",
        output_file_path=output_path,
    )
    assert args[-1] == str(output_path)


def test_build_output_video_file_path_uses_zero_padded_index():
    segment = make_segment_at(timedelta(minutes=5), timedelta(minutes=5, seconds=25), index=3)
    result = build_output_video_file_path(Path("output"), segment)
    assert result == Path("output/segment_03.mp4")


def test_build_output_video_file_path_uses_mp4_extension():
    segment = make_segment_at(timedelta(minutes=5), timedelta(minutes=5, seconds=25))
    result = build_output_video_file_path(Path("output"), segment)
    assert result.suffix == ".mp4"


def test_build_output_subtitle_file_path_uses_srt_extension():
    segment = make_segment_at(timedelta(minutes=5), timedelta(minutes=5, seconds=25))
    result = build_output_subtitle_file_path(Path("output"), segment)
    assert result.suffix == ".srt"


def test_build_output_subtitle_file_path_matches_video_file_name():
    segment = make_segment_at(timedelta(minutes=5), timedelta(minutes=5, seconds=25), index=2)
    video_path = build_output_video_file_path(Path("output"), segment)
    subtitle_path = build_output_subtitle_file_path(Path("output"), segment)
    assert video_path.stem == subtitle_path.stem
