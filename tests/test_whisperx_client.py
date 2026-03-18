"""
tests/test_whisperx_client.py
Tests for api/whisperx_client.py

Uses respx to mock HTTP calls — never hits the real WhisperX API.
"""

import pytest
import respx
import httpx
from pathlib import Path

from api.whisperx_client import WhisperXClient, WhisperXTranscriptionError


FAKE_API_URL = "https://fake-whisperx.example.com"
FAKE_API_KEY = "test-api-key-12345"
SAMPLE_SRT_CONTENT = "1\n00:00:01,000 --> 00:00:05,000\nHello sermon world.\n"


@pytest.fixture
def whisperx_client() -> WhisperXClient:
    return WhisperXClient(api_base_url=FAKE_API_URL, api_key=FAKE_API_KEY)


@pytest.fixture
def fake_audio_file(tmp_path: Path) -> Path:
    audio_file = tmp_path / "sermon_audio.m4a"
    audio_file.write_bytes(b"fake audio content for testing")
    return audio_file


@respx.mock
def test_transcribe_audio_file_returns_srt_when_api_responds_synchronously(
    whisperx_client: WhisperXClient,
    fake_audio_file: Path,
):
    respx.post(f"{FAKE_API_URL}/transcribe").mock(
        return_value=httpx.Response(200, json={"srt": SAMPLE_SRT_CONTENT})
    )

    result = whisperx_client.transcribe_audio_file(fake_audio_file)
    assert result == SAMPLE_SRT_CONTENT


@respx.mock
def test_transcribe_audio_file_polls_until_job_is_complete(
    whisperx_client: WhisperXClient,
    fake_audio_file: Path,
):
    respx.post(f"{FAKE_API_URL}/transcribe").mock(
        return_value=httpx.Response(200, json={"job_id": "job-abc-123"})
    )
    respx.get(f"{FAKE_API_URL}/transcribe/job-abc-123").mock(
        return_value=httpx.Response(200, json={"status": "completed", "srt": SAMPLE_SRT_CONTENT})
    )

    result = whisperx_client.transcribe_audio_file(fake_audio_file)
    assert result == SAMPLE_SRT_CONTENT


@respx.mock
def test_transcribe_audio_file_raises_when_job_fails(
    whisperx_client: WhisperXClient,
    fake_audio_file: Path,
):
    respx.post(f"{FAKE_API_URL}/transcribe").mock(
        return_value=httpx.Response(200, json={"job_id": "job-fail-999"})
    )
    respx.get(f"{FAKE_API_URL}/transcribe/job-fail-999").mock(
        return_value=httpx.Response(200, json={"status": "failed", "error": "Out of memory"})
    )

    with pytest.raises(WhisperXTranscriptionError, match="failed"):
        whisperx_client.transcribe_audio_file(fake_audio_file)


@respx.mock
def test_transcribe_audio_file_raises_when_response_has_no_job_id_or_srt(
    whisperx_client: WhisperXClient,
    fake_audio_file: Path,
):
    respx.post(f"{FAKE_API_URL}/transcribe").mock(
        return_value=httpx.Response(200, json={"unexpected_field": "unexpected_value"})
    )

    with pytest.raises(WhisperXTranscriptionError, match="neither"):
        whisperx_client.transcribe_audio_file(fake_audio_file)


def test_transcribe_audio_file_raises_when_audio_file_does_not_exist(
    whisperx_client: WhisperXClient,
):
    with pytest.raises(FileNotFoundError):
        whisperx_client.transcribe_audio_file(Path("/nonexistent/path/audio.m4a"))


@respx.mock
def test_transcribe_audio_file_sends_bearer_token_in_authorization_header(
    whisperx_client: WhisperXClient,
    fake_audio_file: Path,
):
    submitted_request = None

    def capture_request(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        submitted_request = request
        return httpx.Response(200, json={"srt": SAMPLE_SRT_CONTENT})

    respx.post(f"{FAKE_API_URL}/transcribe").mock(side_effect=capture_request)

    whisperx_client.transcribe_audio_file(fake_audio_file)
    assert submitted_request.headers["Authorization"] == f"Bearer {FAKE_API_KEY}"
