"""Tests for pipeline/audio_track_discovery.py"""

from pathlib import Path

from pipeline.audio_track_discovery import (
    discover_audio_tracks,
    format_discovered_tracks_summary,
)


def test_discover_audio_tracks_finds_latvian_and_russian(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    audio_raw = tmp_path / "Audio RAW"
    audio_raw.mkdir()
    (audio_raw / "sermon_A03.m4a").touch()
    (audio_raw / "sermon_A04.m4a").touch()

    tracks = discover_audio_tracks(video_file)

    assert "lv" in tracks
    assert "ru" in tracks
    assert tracks["lv"].name == "sermon_A03.m4a"
    assert tracks["ru"].name == "sermon_A04.m4a"


def test_discover_audio_tracks_finds_english(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    audio_raw = tmp_path / "Audio RAW"
    audio_raw.mkdir()
    (audio_raw / "sermon_A05.wav").touch()

    tracks = discover_audio_tracks(video_file)

    assert "en" in tracks
    assert tracks["en"].name == "sermon_A05.wav"


def test_discover_audio_tracks_returns_empty_when_no_audio_raw_folder(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    tracks = discover_audio_tracks(video_file)

    assert tracks == {}


def test_discover_audio_tracks_returns_empty_when_no_matching_files(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    audio_raw = tmp_path / "Audio RAW"
    audio_raw.mkdir()
    (audio_raw / "random_file.txt").touch()

    tracks = discover_audio_tracks(video_file)

    assert tracks == {}


def test_discover_audio_tracks_ignores_non_audio_extensions(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    audio_raw = tmp_path / "Audio RAW"
    audio_raw.mkdir()
    (audio_raw / "sermon_A03.txt").touch()
    (audio_raw / "sermon_A04.pdf").touch()

    tracks = discover_audio_tracks(video_file)

    assert tracks == {}


def test_discover_audio_tracks_supports_all_audio_extensions(tmp_path):
    video_file = tmp_path / "sermon.mp4"
    video_file.touch()

    audio_raw = tmp_path / "Audio RAW"
    audio_raw.mkdir()
    (audio_raw / "sermon_A03.flac").touch()

    tracks = discover_audio_tracks(video_file)

    assert "lv" in tracks


def test_format_discovered_tracks_summary_with_tracks():
    tracks = {
        "lv": Path("/some/sermon_A03.m4a"),
        "ru": Path("/some/sermon_A04.m4a"),
    }

    summary = format_discovered_tracks_summary(tracks)

    assert "Latvian" in summary
    assert "Russian" in summary
    assert "_A03" in summary
    assert "_A04" in summary


def test_format_discovered_tracks_summary_empty():
    summary = format_discovered_tracks_summary({})

    assert "No audio tracks found" in summary
