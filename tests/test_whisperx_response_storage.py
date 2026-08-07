"""Tests for pipeline/whisperx_response_storage.py"""

import json
from pathlib import Path

from pipeline.whisperx_response_storage import (
    find_latest_version_number,
    build_version_directory_path,
    build_next_version_directory_path,
    save_whisperx_response,
    load_whisperx_response,
    list_stored_audio_stems_for_version,
)


def test_find_latest_version_number_returns_none_when_no_whisperx_folder(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    assert find_latest_version_number(video_file) is None


def test_find_latest_version_number_returns_none_when_empty_folder(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()
    (tmp_path / "Audio TRANSCRIPT" / "whisperx").mkdir(parents=True)

    assert find_latest_version_number(video_file) is None


def test_find_latest_version_number_finds_single_version(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()
    (tmp_path / "Audio TRANSCRIPT" / "whisperx" / "v1").mkdir(parents=True)

    assert find_latest_version_number(video_file) == 1


def test_find_latest_version_number_finds_highest_version(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()
    whisperx_directory = tmp_path / "Audio TRANSCRIPT" / "whisperx"
    (whisperx_directory / "v1").mkdir(parents=True)
    (whisperx_directory / "v2").mkdir()
    (whisperx_directory / "v3").mkdir()

    assert find_latest_version_number(video_file) == 3


def test_find_latest_version_number_ignores_non_version_folders(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()
    whisperx_directory = tmp_path / "Audio TRANSCRIPT" / "whisperx"
    (whisperx_directory / "v1").mkdir(parents=True)
    (whisperx_directory / "temp").mkdir()
    (whisperx_directory / "notes.txt").touch()

    assert find_latest_version_number(video_file) == 1


def test_build_version_directory_path(tmp_path):
    video_file = tmp_path / "sermon.mp4"

    result = build_version_directory_path(video_file, 3)

    assert result == tmp_path / "Audio TRANSCRIPT" / "whisperx" / "v3"


def test_build_next_version_directory_path_starts_at_v1(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    result = build_next_version_directory_path(video_file)

    assert result == tmp_path / "Audio TRANSCRIPT" / "whisperx" / "v1"


def test_build_next_version_directory_path_increments(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()
    (tmp_path / "Audio TRANSCRIPT" / "whisperx" / "v2").mkdir(parents=True)

    result = build_next_version_directory_path(video_file)

    assert result == tmp_path / "Audio TRANSCRIPT" / "whisperx" / "v3"


def test_save_and_load_whisperx_response(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()
    version_directory = tmp_path / "Audio TRANSCRIPT" / "whisperx" / "v1"

    sample_response = {
        "segments": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Hello world",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.95},
                    {"word": "world", "start": 0.6, "end": 1.0, "score": 0.92},
                ],
            }
        ]
    }

    saved_path = save_whisperx_response(
        video_file_path=video_file,
        audio_file_stem="sermon_A03",
        response_data=sample_response,
        version_directory=version_directory,
    )

    assert saved_path.exists()
    assert saved_path.name == "sermon_A03.json"

    loaded = load_whisperx_response(video_file, "sermon_A03", 1)
    assert loaded == sample_response


def test_list_stored_audio_stems_for_version(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()
    version_directory = tmp_path / "Audio TRANSCRIPT" / "whisperx" / "v1"
    version_directory.mkdir(parents=True)
    (version_directory / "sermon_A03.json").write_text("{}")
    (version_directory / "sermon_A04.json").write_text("{}")

    stems = list_stored_audio_stems_for_version(video_file, 1)

    assert sorted(stems) == ["sermon_A03", "sermon_A04"]


def test_list_stored_audio_stems_returns_empty_when_no_version(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    stems = list_stored_audio_stems_for_version(video_file, 1)

    assert stems == []
