"""
tests/test_audio_cutter.py

Tests for audio segment cutting and output path building.
"""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from models.video_segment import VideoSegment
from pipeline.audio_cutter import (
    cut_audio_segment,
    build_output_audio_file_path,
)


@pytest.fixture
def sample_segment() -> VideoSegment:
    return VideoSegment(
        index=1,
        start_time=timedelta(minutes=5),
        end_time=timedelta(minutes=5, seconds=30),
        transcript_text="God is good.",
        selection_reason="Strong emotional moment",
        suggested_title="God Is Good",
    )


@pytest.fixture
def video_file_path(tmp_path: Path) -> Path:
    return tmp_path / "2026-01-04 Sunday Sermon.mp4"


class TestBuildOutputAudioFilePath:

    def test_includes_language_code_in_filename(
        self, sample_segment: VideoSegment, video_file_path: Path
    ):
        result = build_output_audio_file_path(
            Path("/clips"), sample_segment, "lv", video_file_path,
        )

        assert "[lv]" in result.name
        assert result.suffix == ".wav"

    def test_includes_date_and_title(
        self, sample_segment: VideoSegment, video_file_path: Path
    ):
        result = build_output_audio_file_path(
            Path("/clips"), sample_segment, "ru", video_file_path,
        )

        assert result.name == "2026-01-04 God Is Good [30s] [ru].wav"

    def test_different_languages_produce_different_paths(
        self, sample_segment: VideoSegment, video_file_path: Path
    ):
        lv_path = build_output_audio_file_path(
            Path("/clips"), sample_segment, "lv", video_file_path,
        )
        ru_path = build_output_audio_file_path(
            Path("/clips"), sample_segment, "ru", video_file_path,
        )

        assert lv_path != ru_path
        assert "[lv]" in lv_path.name
        assert "[ru]" in ru_path.name

    def test_works_without_video_file_path(self, sample_segment: VideoSegment):
        result = build_output_audio_file_path(
            Path("/clips"), sample_segment, "en",
        )

        assert result.name == "God Is Good [30s] [en].wav"


class TestCutAudioSegment:

    @patch("pipeline.audio_cutter.run_ffmpeg_command")
    def test_calls_ffmpeg_with_correct_timestamps(
        self,
        mock_ffmpeg: MagicMock,
        sample_segment: VideoSegment,
        tmp_path: Path,
    ):
        source_audio = tmp_path / "sermon.offset.wav"
        source_audio.write_bytes(b"\x00" * 100)
        output_path = tmp_path / "clip.wav"

        cut_audio_segment(source_audio, sample_segment, output_path)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-ss" in ffmpeg_arguments
        assert "-to" in ffmpeg_arguments
        assert "-c" in ffmpeg_arguments
        assert "copy" in ffmpeg_arguments
        assert str(source_audio) in ffmpeg_arguments
        assert str(output_path) in ffmpeg_arguments

    @patch("pipeline.audio_cutter.run_ffmpeg_command")
    def test_creates_output_directory(
        self,
        mock_ffmpeg: MagicMock,
        sample_segment: VideoSegment,
        tmp_path: Path,
    ):
        source_audio = tmp_path / "sermon.offset.wav"
        source_audio.write_bytes(b"\x00" * 100)
        nested_output = tmp_path / "clips" / "v2" / "clip.wav"

        cut_audio_segment(source_audio, sample_segment, nested_output)

        assert nested_output.parent.exists()

    @patch("pipeline.audio_cutter.run_ffmpeg_command")
    def test_returns_output_path(
        self,
        mock_ffmpeg: MagicMock,
        sample_segment: VideoSegment,
        tmp_path: Path,
    ):
        source_audio = tmp_path / "sermon.offset.wav"
        source_audio.write_bytes(b"\x00" * 100)
        output_path = tmp_path / "clip.wav"

        result = cut_audio_segment(source_audio, sample_segment, output_path)

        assert result == output_path
