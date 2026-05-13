"""
api/manus_client.py

HTTP wrapper for the Manus AI agent API.
Submits a task via POST /v1/tasks, polls GET /v1/tasks/{id} until complete.
Extracts text from the output array structure.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

MANUS_API_BASE_URL = "https://api.manus.ai/v1"
POLLING_INTERVAL_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 60
MAX_POLLING_WAIT_SECONDS = 900  # 15 minutes


class ManusTaskError(Exception):
    pass


class ManusClient:
    def __init__(self, api_key: str) -> None:
        self.request_headers = {
            "API_KEY": api_key,
            "Content-Type": "application/json",
        }
        self.last_task_id: Optional[str] = None

    def submit_prompt_and_wait_for_response(
        self,
        prompt_text: str,
        on_progress: Optional[Callable[[str], None]] = None,
        on_task_created: Optional[Callable[[str], None]] = None,
        project_id: Optional[str] = None,
        task_title: Optional[str] = None,
    ) -> str:
        """
        Submit a prompt to Manus AI and block until the task is complete.
        Returns the raw text response from the agent.
        Calls on_task_created(task_id) immediately after submission,
        before polling begins — use this to persist the task_id for resume.
        """
        task_id = self._submit_task(prompt_text, on_progress, project_id)
        self.last_task_id = task_id
        if on_task_created:
            on_task_created(task_id)
        return self._poll_until_task_complete(task_id, on_progress, task_title)

    def fetch_task_response(
        self,
        task_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Fetch response from an existing Manus task by ID.
        If the task is still running, resumes polling until complete.
        """
        def _log(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        self.last_task_id = task_id
        _log(f"Fetching task: {task_id}")

        try:
            status_response = httpx.get(
                url=f"{MANUS_API_BASE_URL}/tasks/{task_id}",
                headers=self.request_headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            status_response.raise_for_status()
            status_data = status_response.json()
        except httpx.HTTPError as error:
            raise ManusTaskError(f"Failed to fetch task {task_id}: {error}")

        task_status = status_data.get("status", "").lower()
        _log(f"Task status: {task_status}")

        if task_status == "completed":
            response_text = self._extract_text_from_output(status_data)
            if not response_text:
                raise ManusTaskError("Task completed but no text found in output")
            return response_text

        if task_status == "failed":
            error_msg = status_data.get("error", "Unknown error")
            raise ManusTaskError(f"Task failed: {error_msg}")

        _log("Task still running — resuming poll...")
        return self._poll_until_task_complete(task_id, on_progress)

    def list_tasks(
        self,
        project_id: Optional[str] = None,
    ) -> list[dict]:
        """List recent tasks from Manus API. Returns empty list on failure."""
        try:
            params: dict = {}
            if project_id:
                params["projectId"] = project_id
            response = httpx.get(
                url=f"{MANUS_API_BASE_URL}/tasks",
                headers=self.request_headers,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            return data.get("tasks", data.get("data", []))
        except httpx.HTTPError as error:
            logger.warning(f"Failed to list Manus tasks: {error}")
            return []

    def _submit_task(
        self,
        prompt_text: str,
        on_progress: Optional[Callable[[str], None]] = None,
        project_id: Optional[str] = None,
    ) -> str:
        def _log(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        _log("Submitting task to Manus AI...")

        request_body = {
            "prompt": prompt_text,
            "agentProfile": "manus-1.6",
            "taskMode": "agent",
            "hideInTaskList": True,
        }
        if project_id:
            request_body["projectId"] = project_id

        response = httpx.post(
            url=f"{MANUS_API_BASE_URL}/tasks",
            headers=self.request_headers,
            json=request_body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_data = response.json()

        task_id = response_data.get("task_id")
        if not task_id:
            raise ManusTaskError(
                f"Manus response did not contain a task_id: {response_data}"
            )

        task_url = response_data.get("task_url", f"https://manus.im/tasks/{task_id}")
        _log(f"Task submitted: {task_id}")
        _log(f"Task URL: {task_url}")
        return task_id

    def _update_task_title(
        self,
        task_id: str,
        title: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        def _log(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        _log(f"Renaming task to: {title}")
        try:
            response = httpx.put(
                url=f"{MANUS_API_BASE_URL}/tasks/{task_id}",
                headers=self.request_headers,
                json={"title": title},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            _log(f"Warning: failed to update task title: {e}")

    def _poll_until_task_complete(
        self,
        task_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
        task_title: Optional[str] = None,
    ) -> str:
        def _log(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        elapsed_seconds = 0
        title_updated = False
        _log(f"Waiting for Manus AI (polling every {POLLING_INTERVAL_SECONDS}s, max {MAX_POLLING_WAIT_SECONDS // 60}min)...")

        while elapsed_seconds < MAX_POLLING_WAIT_SECONDS:
            time.sleep(POLLING_INTERVAL_SECONDS)
            elapsed_seconds += POLLING_INTERVAL_SECONDS
            remaining = MAX_POLLING_WAIT_SECONDS - elapsed_seconds

            try:
                status_response = httpx.get(
                    url=f"{MANUS_API_BASE_URL}/tasks/{task_id}",
                    headers=self.request_headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                status_response.raise_for_status()
                status_data = status_response.json()
            except httpx.HTTPError as e:
                _log(f"Poll error ({elapsed_seconds}s elapsed, {remaining}s left): {e}")
                continue

            # Rename task on first successful poll
            if task_title and not title_updated:
                self._update_task_title(task_id, task_title, on_progress)
                title_updated = True

            task_status = status_data.get("status", "").lower()
            _log(f"Status: {task_status} ({elapsed_seconds}s elapsed, {remaining}s left)")

            if task_status == "completed":
                response_text = self._extract_text_from_output(status_data)
                if not response_text:
                    raise ManusTaskError(
                        f"Task completed but no text found in output: {status_data}"
                    )
                _log(f"Task complete ({elapsed_seconds}s)")
                return response_text

            if task_status == "failed":
                error_msg = status_data.get("error", "Unknown error")
                raise ManusTaskError(f"Task failed: {error_msg}")

        raise ManusTaskError(
            f"Task {task_id} did not complete within {MAX_POLLING_WAIT_SECONDS}s"
        )

    def _extract_text_from_output(self, task_data: dict) -> str:
        """
        Extract text from Manus output structure.
        output is a list of message objects. We want the LAST assistant
        message's text content (that's the actual response, not the echoed prompt).
        Also downloads file attachments that Manus may return instead of inline text.
        """
        output = task_data.get("output")
        if not output:
            return ""

        if isinstance(output, str):
            return output

        last_text = ""
        file_urls: list[str] = []

        for message in output:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not content:
                continue

            msg_text_parts = []
            if isinstance(content, str):
                msg_text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            msg_text_parts.append(text)
                        file_url = item.get("url") or item.get("file_url") or item.get("fileUrl")
                        if file_url:
                            file_urls.append(file_url)
                    elif isinstance(item, str):
                        msg_text_parts.append(item)

            for artifact in message.get("artifacts", []):
                if isinstance(artifact, dict):
                    artifact_url = artifact.get("url") or artifact.get("file_url") or artifact.get("fileUrl")
                    if artifact_url:
                        file_urls.append(artifact_url)
                    artifact_content = artifact.get("content", "")
                    if artifact_content:
                        msg_text_parts.append(artifact_content)

            combined = "\n".join(msg_text_parts).strip()
            if combined:
                last_text = combined

        for artifact in task_data.get("artifacts", []):
            if isinstance(artifact, dict):
                artifact_url = artifact.get("url") or artifact.get("file_url") or artifact.get("fileUrl")
                if artifact_url:
                    file_urls.append(artifact_url)
                artifact_content = artifact.get("content", "")
                if artifact_content:
                    last_text = f"{last_text}\n{artifact_content}" if last_text else artifact_content

        for file_url in file_urls:
            downloaded_content = self._download_file_content(file_url)
            if downloaded_content:
                last_text = f"{last_text}\n{downloaded_content}" if last_text else downloaded_content

        return last_text

    def _download_file_content(self, file_url: str) -> str:
        """Download content from a file URL attached to a Manus task response."""
        try:
            logger.info(f"Downloading file attachment: {file_url}")
            response = httpx.get(
                url=file_url,
                headers=self.request_headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            pass
        try:
            response = httpx.get(
                url=file_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as error:
            logger.warning(f"Failed to download file attachment {file_url}: {error}")
            return ""
