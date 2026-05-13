"""
tests/test_manus_client.py
Tests for api/manus_client.py

Uses respx to mock HTTP calls — never hits the real Manus API.
"""

import pytest
import respx
import httpx

from api.manus_client import ManusClient, ManusTaskError, MANUS_API_BASE_URL


FAKE_API_KEY = "manus-test-key-99999"
SAMPLE_PROMPT = "Select the best sermon segments from this SRT file."
SAMPLE_RESPONSE_TEXT = '{"title": "Test", "segments": []}'

COMPLETED_TASK_WITH_OUTPUT = {
    "status": "completed",
    "output": [
        {
            "content": [
                {"text": SAMPLE_RESPONSE_TEXT}
            ]
        }
    ],
}

COMPLETED_TASK_EMPTY_OUTPUT = {
    "status": "completed",
    "output": [],
}


@pytest.fixture
def manus_client() -> ManusClient:
    return ManusClient(api_key=FAKE_API_KEY)


@respx.mock
def test_submit_prompt_returns_response_text_when_task_completes(manus_client: ManusClient):
    respx.post(f"{MANUS_API_BASE_URL}/tasks").mock(
        return_value=httpx.Response(200, json={"task_id": "task-xyz-001"})
    )
    respx.get(f"{MANUS_API_BASE_URL}/tasks/task-xyz-001").mock(
        return_value=httpx.Response(200, json=COMPLETED_TASK_WITH_OUTPUT)
    )

    result = manus_client.submit_prompt_and_wait_for_response(SAMPLE_PROMPT)
    assert result == SAMPLE_RESPONSE_TEXT


@respx.mock
def test_submit_prompt_raises_when_task_id_missing_from_response(manus_client: ManusClient):
    respx.post(f"{MANUS_API_BASE_URL}/tasks").mock(
        return_value=httpx.Response(200, json={"unexpected": "no task id here"})
    )

    with pytest.raises(ManusTaskError, match="task_id"):
        manus_client.submit_prompt_and_wait_for_response(SAMPLE_PROMPT)


@respx.mock
def test_submit_prompt_raises_when_task_fails(manus_client: ManusClient):
    respx.post(f"{MANUS_API_BASE_URL}/tasks").mock(
        return_value=httpx.Response(200, json={"task_id": "task-fail-002"})
    )
    respx.get(f"{MANUS_API_BASE_URL}/tasks/task-fail-002").mock(
        return_value=httpx.Response(
            200,
            json={"status": "failed", "error": "Rate limit exceeded"},
        )
    )

    with pytest.raises(ManusTaskError, match="failed"):
        manus_client.submit_prompt_and_wait_for_response(SAMPLE_PROMPT)


@respx.mock
def test_submit_prompt_raises_when_completed_response_has_no_result(manus_client: ManusClient):
    respx.post(f"{MANUS_API_BASE_URL}/tasks").mock(
        return_value=httpx.Response(200, json={"task_id": "task-empty-003"})
    )
    respx.get(f"{MANUS_API_BASE_URL}/tasks/task-empty-003").mock(
        return_value=httpx.Response(200, json=COMPLETED_TASK_EMPTY_OUTPUT)
    )

    with pytest.raises(ManusTaskError, match="no text found"):
        manus_client.submit_prompt_and_wait_for_response(SAMPLE_PROMPT)


@respx.mock
def test_submit_prompt_sends_api_key_in_header(manus_client: ManusClient):
    submitted_request = None

    def capture_request(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_request
        submitted_request = request
        return httpx.Response(200, json={"task_id": "task-header-check"})

    respx.post(f"{MANUS_API_BASE_URL}/tasks").mock(side_effect=capture_request)
    respx.get(f"{MANUS_API_BASE_URL}/tasks/task-header-check").mock(
        return_value=httpx.Response(200, json=COMPLETED_TASK_WITH_OUTPUT)
    )

    manus_client.submit_prompt_and_wait_for_response(SAMPLE_PROMPT)
    assert submitted_request.headers["API_KEY"] == FAKE_API_KEY


@respx.mock
def test_submit_prompt_includes_prompt_in_request_body(manus_client: ManusClient):
    import json

    submitted_body = None

    def capture_request(request: httpx.Request) -> httpx.Response:
        nonlocal submitted_body
        submitted_body = json.loads(request.content)
        return httpx.Response(200, json={"task_id": "task-body-check"})

    respx.post(f"{MANUS_API_BASE_URL}/tasks").mock(side_effect=capture_request)
    respx.get(f"{MANUS_API_BASE_URL}/tasks/task-body-check").mock(
        return_value=httpx.Response(200, json=COMPLETED_TASK_WITH_OUTPUT)
    )

    manus_client.submit_prompt_and_wait_for_response(SAMPLE_PROMPT)
    assert submitted_body["prompt"] == SAMPLE_PROMPT
