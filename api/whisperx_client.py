"""
api/whisperx_client.py

HTTP wrapper for the WhisperX REST transcription service.
Uses TUS protocol for chunked uploads (bypasses Cloudflare body size limits).
Upload auto-triggers transcription, then polls /task/{id} for results.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

CHUNK_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB chunks
POLLING_INTERVAL_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 120
UPLOAD_CHUNK_TIMEOUT_SECONDS = 300
MAX_POLLING_WAIT_SECONDS = 3600


class WhisperXTranscriptionError(Exception):
    pass


class WhisperXClient:
    def __init__(self, api_base_url: str, api_key: str) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.auth_headers = {"Authorization": f"Bearer {api_key}"}

    def transcribe_audio_file(
        self,
        audio_file_path: Path,
        language_code: str = "lv",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Upload audio via TUS chunked protocol (auto-triggers transcription),
        then poll until complete. Returns SRT text.
        """
        task_id = self._upload_via_tus(audio_file_path, language_code, on_progress)
        return self._poll_until_complete(task_id, on_progress)

    def transcribe_audio_file_raw(
        self,
        audio_file_path: Path,
        language_code: str = "lv",
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        Upload audio via TUS chunked protocol (auto-triggers transcription),
        then poll until complete. Returns the full result dict including
        word-level data (segments[].words[]).
        """
        task_id = self._upload_via_tus(audio_file_path, language_code, on_progress)
        return self._poll_until_complete_raw(task_id, on_progress)

    def _upload_via_tus(
        self,
        audio_file_path: Path,
        language_code: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        def _log(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        file_size = audio_file_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        filename = audio_file_path.name

        _log(f"Uploading {filename} ({file_size_mb:.1f} MB) via chunked upload...")

        # Step 1: Create TUS upload with metadata
        fn_b64 = base64.b64encode(filename.encode()).decode()
        lang_b64 = base64.b64encode(language_code.encode()).decode()
        model_b64 = base64.b64encode(b"large-v3").decode()
        metadata = f"filename {fn_b64},language {lang_b64},model {model_b64}"

        create_resp = httpx.post(
            f"{self.api_base_url}/uploads/files/",
            headers={
                **self.auth_headers,
                "Tus-Resumable": "1.0.0",
                "Upload-Length": str(file_size),
                "Upload-Metadata": metadata,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if create_resp.status_code not in (200, 201):
            raise WhisperXTranscriptionError(
                f"TUS create failed ({create_resp.status_code}): {create_resp.text}"
            )

        upload_url = create_resp.headers.get("location", "")
        if not upload_url:
            raise WhisperXTranscriptionError("TUS create response missing Location header")

        _log(f"Upload session created, sending data in {CHUNK_SIZE_BYTES // (1024*1024)}MB chunks...")

        # Step 2: Upload file in chunks via PATCH
        upload_start = time.time()
        offset = 0

        with open(audio_file_path, "rb") as f:
            while offset < file_size:
                chunk = f.read(CHUNK_SIZE_BYTES)
                chunk_num = (offset // CHUNK_SIZE_BYTES) + 1
                total_chunks = (file_size + CHUNK_SIZE_BYTES - 1) // CHUNK_SIZE_BYTES
                _log(f"Uploading chunk {chunk_num}/{total_chunks} ({offset / (1024*1024):.0f}/{file_size_mb:.0f} MB)...")

                patch_resp = httpx.patch(
                    upload_url,
                    headers={
                        **self.auth_headers,
                        "Tus-Resumable": "1.0.0",
                        "Upload-Offset": str(offset),
                        "Content-Type": "application/offset+octet-stream",
                    },
                    content=chunk,
                    timeout=UPLOAD_CHUNK_TIMEOUT_SECONDS,
                )

                if patch_resp.status_code not in (200, 204):
                    raise WhisperXTranscriptionError(
                        f"TUS upload chunk failed ({patch_resp.status_code}): {patch_resp.text}"
                    )

                new_offset = int(patch_resp.headers.get("upload-offset", offset + len(chunk)))
                offset = new_offset

        upload_elapsed = time.time() - upload_start
        _log(f"Upload complete ({upload_elapsed:.0f}s)")

        # Step 3: Find the task ID created by the upload
        task_id = self._find_latest_task_id(filename)
        _log(f"Transcription task: {task_id}")
        return task_id

    def _find_latest_task_id(self, filename: str) -> str:
        """Find the most recent task matching our filename."""
        time.sleep(2)  # brief wait for task creation
        resp = httpx.get(
            f"{self.api_base_url}/task/all",
            headers=self.auth_headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()

        tasks = data.get("tasks", data) if isinstance(data, dict) else data
        if not isinstance(tasks, list):
            raise WhisperXTranscriptionError(f"Unexpected /task/all response: {data}")

        # Find most recent task with matching filename
        for task in reversed(tasks):
            if task.get("file_name") == filename:
                return task["identifier"]

        # Fallback: just use the most recent task
        if tasks:
            return tasks[-1]["identifier"]

        raise WhisperXTranscriptionError("No tasks found after upload")

    def _poll_until_complete_raw(
        self,
        task_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Poll until task completes, return the full result dict."""
        def _log(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        elapsed_seconds = 0
        _log("Waiting for transcription to complete...")

        while elapsed_seconds < MAX_POLLING_WAIT_SECONDS:
            time.sleep(POLLING_INTERVAL_SECONDS)
            elapsed_seconds += POLLING_INTERVAL_SECONDS
            remaining = MAX_POLLING_WAIT_SECONDS - elapsed_seconds

            # Check progress
            try:
                progress_resp = httpx.get(
                    f"{self.api_base_url}/tasks/{task_id}/progress",
                    headers=self.auth_headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if progress_resp.status_code == 200:
                    progress_data = progress_resp.json()
                    percentage = progress_data.get("progress_percentage", "?")
                    stage = progress_data.get("progress_stage", "")
                    _log(f"Progress: {percentage}% -- {stage} ({elapsed_seconds}s elapsed, {remaining}s left)")
            except Exception:
                _log(f"Polling... ({elapsed_seconds}s elapsed, {remaining}s left)")

            # Check task result
            try:
                task_resp = httpx.get(
                    f"{self.api_base_url}/task/{task_id}",
                    headers=self.auth_headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                task_resp.raise_for_status()
                task_data = task_resp.json()
            except httpx.HTTPError as error:
                _log(f"Poll error: {error}")
                continue

            task_status = str(task_data.get("status", "")).lower()

            if task_status in ("completed", "done", "finished", "success"):
                result = task_data.get("result")
                if not result:
                    raise WhisperXTranscriptionError(
                        f"Task completed but no result: {task_data}"
                    )
                _log(f"Transcription complete ({elapsed_seconds}s)")
                return result

            if task_status in ("failed", "error"):
                error_msg = task_data.get("error", "Unknown error")
                raise WhisperXTranscriptionError(f"Task failed: {error_msg}")

        raise WhisperXTranscriptionError(
            f"Task {task_id} did not complete within {MAX_POLLING_WAIT_SECONDS}s"
        )

    def _poll_until_complete(
        self,
        task_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Poll until task completes, return SRT text (backward compatible)."""
        result = self._poll_until_complete_raw(task_id, on_progress)
        return self._result_to_srt(result)

    @staticmethod
    def _result_to_srt(result) -> str:
        """Convert WhisperX result (segments list or dict) to SRT format."""
        if isinstance(result, str):
            return result

        segments = result if isinstance(result, list) else result.get("segments", [])

        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "").strip()
            if not text:
                continue
            srt_lines.append(str(i))
            srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            srt_lines.append(text)
            srt_lines.append("")

        return "\n".join(srt_lines)


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
