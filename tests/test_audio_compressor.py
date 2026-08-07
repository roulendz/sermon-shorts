"""
tests/test_audio_compressor.py

Tests for audio compression and offset prepending logic.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.audio_compressor import (
    _build_adelay_filter_value,
    _convert_audio,
    compress_audio_for_transkriptor,
    prepare_audio_with_offset,
    prepare_original_audio_with_offset,
    DEFAULT_AUDIO_OFFSET_SECONDS,
    WHISPERX_CODEC_ARGUMENTS,
    WHISPERX_FORMAT_LABEL,
    TRANSKRIPTOR_CODEC_ARGUMENTS,
    TRANSKRIPTOR_FORMAT_LABEL,
)


def _create_output_file_side_effect(ffmpeg_arguments, output_file_path):
    output_file_path.write_bytes(b"\x00" * 256)


class TestBuildAdelayFilterValue:

    def test_converts_seconds_to_millisecond_delay(self):
        result = _build_adelay_filter_value(9.13)
        assert result == "adelay=9130|9130"

    def test_handles_zero_offset(self):
        result = _build_adelay_filter_value(0.0)
        assert result == "adelay=0|0"

    def test_handles_fractional_seconds(self):
        result = _build_adelay_filter_value(2.5)
        assert result == "adelay=2500|2500"


class TestCompressAudioForTranskriptor:

    @pytest.fixture
    def audio_file(self, tmp_path: Path) -> Path:
        audio_path = tmp_path / "sermon_A03.wav"
        audio_path.write_bytes(b"\x00" * 1024)
        return audio_path

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_no_adelay_filter_applied(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        compress_audio_for_transkriptor(audio_file)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-af" not in ffmpeg_arguments

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_output_path_is_transkriptor_mp3(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        result = compress_audio_for_transkriptor(audio_file)

        assert result.name == "sermon_A03.transkriptor.mp3"

    def test_returns_cached_file_when_exists(self, audio_file: Path):
        cached_path = audio_file.parent / "sermon_A03.transkriptor.mp3"
        cached_path.write_bytes(b"\x00" * 512)

        result = compress_audio_for_transkriptor(audio_file)

        assert result == cached_path

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_uses_mp3_codec_and_mono_settings(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        compress_audio_for_transkriptor(audio_file)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-acodec" in ffmpeg_arguments
        codec_index = ffmpeg_arguments.index("-acodec") + 1
        assert ffmpeg_arguments[codec_index] == "libmp3lame"
        assert "-ac" in ffmpeg_arguments
        channels_index = ffmpeg_arguments.index("-ac") + 1
        assert ffmpeg_arguments[channels_index] == "1"

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_calls_on_progress_callback(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        progress_messages = []
        compress_audio_for_transkriptor(
            audio_file, on_progress=progress_messages.append
        )

        assert len(progress_messages) >= 2
        assert any("Converting" in message for message in progress_messages)
        assert any("Converted" in message for message in progress_messages)


class TestPrepareOriginalAudioWithOffset:

    @pytest.fixture
    def audio_file(self, tmp_path: Path) -> Path:
        audio_path = tmp_path / "sermon_A03.wav"
        audio_path.write_bytes(b"\x00" * 1024)
        return audio_path

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_preserves_original_sample_rate(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        prepare_original_audio_with_offset(audio_file)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-ar" not in ffmpeg_arguments
        assert "-ac" not in ffmpeg_arguments

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_output_path_is_offset_wav(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        result = prepare_original_audio_with_offset(audio_file)

        assert result.name == "sermon_A03.offset.wav"

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_includes_adelay_filter(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        prepare_original_audio_with_offset(audio_file)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-af" in ffmpeg_arguments
        adelay_index = ffmpeg_arguments.index("-af") + 1
        assert ffmpeg_arguments[adelay_index] == "adelay=9130|9130"

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_uses_lossless_pcm_codec(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        prepare_original_audio_with_offset(audio_file)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-acodec" in ffmpeg_arguments
        codec_index = ffmpeg_arguments.index("-acodec") + 1
        assert ffmpeg_arguments[codec_index] == "pcm_s16le"


class TestPrepareAudioWithOffset:

    @pytest.fixture
    def audio_file(self, tmp_path: Path) -> Path:
        audio_path = tmp_path / "sermon_A03.wav"
        audio_path.write_bytes(b"\x00" * 1024)
        return audio_path

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_includes_adelay_filter(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        prepare_audio_with_offset(audio_file, offset_seconds=9.13)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-af" in ffmpeg_arguments
        adelay_index = ffmpeg_arguments.index("-af") + 1
        assert ffmpeg_arguments[adelay_index] == "adelay=9130|9130"

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_output_path_is_wav(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        result = prepare_audio_with_offset(audio_file)

        assert result.name == "sermon_A03.offset.16k.wav"

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_uses_pcm_wav_codec(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        prepare_audio_with_offset(audio_file)

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-acodec" in ffmpeg_arguments
        codec_index = ffmpeg_arguments.index("-acodec") + 1
        assert ffmpeg_arguments[codec_index] == "pcm_s16le"


class TestSingleSourceOfTruthFlow:
    """
    Verifies that WhisperX and Transkriptor share the same offset audio
    by chaining prepare_audio_with_offset -> compress_audio_for_transkriptor.
    """

    @pytest.fixture
    def audio_file(self, tmp_path: Path) -> Path:
        audio_path = tmp_path / "sermon_A03.wav"
        audio_path.write_bytes(b"\x00" * 1024)
        return audio_path

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_transkriptor_mp3_derives_from_offset_wav(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        offset_wav = prepare_audio_with_offset(audio_file)
        transkriptor_mp3 = compress_audio_for_transkriptor(offset_wav)

        assert mock_ffmpeg.call_count == 2

        wav_args = mock_ffmpeg.call_args_list[0][0][0]
        assert "adelay=9130|9130" in wav_args
        assert "pcm_s16le" in wav_args

        mp3_args = mock_ffmpeg.call_args_list[1][0][0]
        assert "-af" not in mp3_args
        assert "libmp3lame" in mp3_args
        assert str(offset_wav) in mp3_args

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_whisperx_and_transkriptor_share_same_offset_wav(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        whisperx_wav = prepare_audio_with_offset(audio_file)
        transkriptor_wav = prepare_audio_with_offset(audio_file)

        assert whisperx_wav == transkriptor_wav
        assert mock_ffmpeg.call_count == 1


class TestConvertAudioSharedEngine:

    @pytest.fixture
    def audio_file(self, tmp_path: Path) -> Path:
        audio_path = tmp_path / "sermon_A03.wav"
        audio_path.write_bytes(b"\x00" * 1024)
        return audio_path

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_wav_and_mp3_share_same_engine(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        wav_result = _convert_audio(
            audio_file,
            output_suffix=".test.wav",
            codec_arguments=WHISPERX_CODEC_ARGUMENTS,
            format_label=WHISPERX_FORMAT_LABEL,
            offset_seconds=9.13,
        )
        mp3_result = _convert_audio(
            audio_file,
            output_suffix=".test.mp3",
            codec_arguments=TRANSKRIPTOR_CODEC_ARGUMENTS,
            format_label=TRANSKRIPTOR_FORMAT_LABEL,
            offset_seconds=9.13,
        )

        assert wav_result.suffix == ".wav"
        assert mp3_result.suffix == ".mp3"
        assert mock_ffmpeg.call_count == 2

        wav_args = mock_ffmpeg.call_args_list[0][0][0]
        mp3_args = mock_ffmpeg.call_args_list[1][0][0]
        assert "pcm_s16le" in wav_args
        assert "libmp3lame" in mp3_args

    @patch("pipeline.audio_compressor._run_ffmpeg_conversion", side_effect=_create_output_file_side_effect)
    def test_skips_adelay_when_offset_is_zero(
        self, mock_ffmpeg: MagicMock, audio_file: Path
    ):
        _convert_audio(
            audio_file,
            output_suffix=".test.wav",
            codec_arguments=WHISPERX_CODEC_ARGUMENTS,
            format_label=WHISPERX_FORMAT_LABEL,
            offset_seconds=0.0,
        )

        ffmpeg_arguments = mock_ffmpeg.call_args[0][0]
        assert "-af" not in ffmpeg_arguments
