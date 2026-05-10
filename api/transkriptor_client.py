"""
api/transkriptor_client.py

HTTP wrapper for the Transkriptor transcription API.
3-step upload: get presigned URL → PUT binary → initiate transcription.
Polls GET /developer/files/{order_id}/content until complete, then exports SRT.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)

TRANSKRIPTOR_API_BASE_URL = "https://api.tor.app/developer"
POLLING_INTERVAL_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 60
UPLOAD_TIMEOUT_SECONDS = 600
MAX_POLLING_WAIT_SECONDS = 3600  # 60 minutes (slower service)


class TranskriptorTranscriptionError(Exception):
    pass


class TranskriptorClient:
    def __init__(self, api_key: str) -> None:
        self.authorization_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def transcribe_audio_file(
        self,
        audio_file_path: Path,
        language_code: str = "lv-LV",
        display_file_name: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Upload audio file, initiate transcription, poll until complete,
        and export as SRT. Returns raw SRT text.
        """
        public_url = self._upload_file(audio_file_path, display_file_name, on_progress)
        order_id = self._initiate_transcription(public_url, language_code, on_progress)
        self._poll_until_complete(order_id, on_progress)
        return self._export_as_srt(order_id, on_progress)

    def _upload_file(
        self,
        audio_file_path: Path,
        display_file_name: Optional[str] = None,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Upload file via presigned URL. Returns the public_url for transcription."""
        def _log(message: str) -> None:
            logger.info(message)
            if on_progress:
                on_progress(message)

        file_extension = audio_file_path.suffix
        file_name = (display_file_name + file_extension) if display_file_name else audio_file_path.name
        file_size_megabytes = audio_file_path.stat().st_size / (1024 * 1024)
        _log(f"Getting upload URL for: {file_name} ({file_size_megabytes:.1f} MB)")

        # Step 1: Get presigned upload URL
        response = httpx.post(
            url=f"{TRANSKRIPTOR_API_BASE_URL}/transcription/local_file/get_upload_url",
            headers=self.authorization_headers,
            json={"file_name": file_name},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_data = response.json()

        upload_url = response_data.get("upload_url")
        public_url = response_data.get("public_url")

        if not upload_url or not public_url:
            raise TranskriptorTranscriptionError(
                f"Missing upload_url or public_url in response: {response_data}"
            )

        _log("Upload URL obtained, uploading file...")

        # Step 2: PUT binary file to presigned URL
        upload_start = time.time()
        with open(audio_file_path, "rb") as file_handle:
            file_content = file_handle.read()

        upload_response = httpx.put(
            url=upload_url,
            content=file_content,
            timeout=UPLOAD_TIMEOUT_SECONDS,
        )

        if upload_response.status_code not in (200, 201):
            raise TranskriptorTranscriptionError(
                f"File upload failed ({upload_response.status_code}): {upload_response.text}"
            )

        upload_elapsed = time.time() - upload_start
        _log(f"File uploaded ({upload_elapsed:.0f}s)")

        return public_url

    def _initiate_transcription(
        self,
        public_url: str,
        language_code: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Start transcription job. Returns order_id for polling."""
        def _log(message: str) -> None:
            logger.info(message)
            if on_progress:
                on_progress(message)

        _log(f"Initiating transcription (language: {language_code})...")

        response = httpx.post(
            url=f"{TRANSKRIPTOR_API_BASE_URL}/transcription/local_file/initiate_transcription",
            headers=self.authorization_headers,
            json={
                "url": public_url,
                "language": language_code,
                "service": "Standard",
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_data = response.json()

        order_id = response_data.get("order_id")
        if not order_id:
            raise TranskriptorTranscriptionError(
                f"No order_id in initiate response: {response_data}"
            )

        _log(f"Transcription initiated: {order_id}")
        return order_id

    def _poll_until_complete(
        self,
        order_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Poll until transcription status is 'Completed'."""
        def _log(message: str) -> None:
            logger.info(message)
            if on_progress:
                on_progress(message)

        elapsed_seconds = 0
        _log(f"Waiting for transcription (polling every {POLLING_INTERVAL_SECONDS}s, max {MAX_POLLING_WAIT_SECONDS // 60}min)...")

        while elapsed_seconds < MAX_POLLING_WAIT_SECONDS:
            time.sleep(POLLING_INTERVAL_SECONDS)
            elapsed_seconds += POLLING_INTERVAL_SECONDS
            remaining = MAX_POLLING_WAIT_SECONDS - elapsed_seconds

            try:
                response = httpx.get(
                    url=f"{TRANSKRIPTOR_API_BASE_URL}/files/{order_id}/content",
                    headers=self.authorization_headers,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                response_data = response.json()
            except httpx.HTTPError as error:
                _log(f"Poll error ({elapsed_seconds}s elapsed, {remaining}s left): {error}")
                continue

            body = response_data.get("body", response_data)
            status = body.get("status", "Unknown")
            _log(f"Status: {status} ({elapsed_seconds}s elapsed, {remaining}s left)")

            if status == "Completed":
                _log(f"Transcription complete ({elapsed_seconds}s)")
                return

            if status in ("Failed", "Error"):
                raise TranskriptorTranscriptionError(
                    f"Transcription failed: {body}"
                )

        raise TranskriptorTranscriptionError(
            f"Order {order_id} did not complete within {MAX_POLLING_WAIT_SECONDS}s"
        )

    def _export_as_srt(
        self,
        order_id: str,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Export completed transcription as SRT format."""
        def _log(message: str) -> None:
            logger.info(message)
            if on_progress:
                on_progress(message)

        _log("Exporting transcription as SRT...")

        response = httpx.post(
            url=f"{TRANSKRIPTOR_API_BASE_URL}/files/{order_id}/content/export",
            headers=self.authorization_headers,
            json={
                "export_type": "srt",
                "include_speaker_names": False,
                "include_timestamps": True,
                "merge_same_speaker_segments": True,
                "is_single_paragraph": False,
                "paragraph_size": 2,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response_data = response.json()

        body = response_data.get("body", response_data)
        srt_content = body.get("content", "")
        presigned_url = body.get("presignedUrl", "")

        if srt_content:
            _log("SRT content received from export")
            return srt_content

        if presigned_url:
            _log("Downloading SRT from presigned URL...")
            download_response = httpx.get(presigned_url, timeout=REQUEST_TIMEOUT_SECONDS)
            download_response.raise_for_status()
            return download_response.text

        raise TranskriptorTranscriptionError(
            f"Export response contained no content or presignedUrl: {response_data}"
        )
