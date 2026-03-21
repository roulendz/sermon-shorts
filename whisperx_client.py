"""
whisperx_client.py
Submits audio to WhisperX API and retrieves SRT transcript.
"""

import time
import logging
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)


class WhisperXClient:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def transcribe(self, audio_path: str, language: str = "en", poll_interval: int = 10) -> str:
        """
        Submit audio file to WhisperX and return SRT string.
        Polls until transcription is complete.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Submitting audio to WhisperX: {audio_path.name} ({audio_path.stat().st_size / 1e6:.1f} MB)")

        # --- Submit job ---
        with open(audio_path, "rb") as f:
            response = httpx.post(
                f"{self.api_url}/transcribe",
                headers=self.headers,
                files={"audio": (audio_path.name, f, "audio/mpeg")},
                data={"language": language, "output_format": "srt"},
                timeout=120,
            )
        response.raise_for_status()
        result = response.json()

        # Handle both sync response (srt returned immediately) and async (job_id)
        if "srt" in result:
            logger.info("WhisperX returned SRT immediately (sync mode)")
            return result["srt"]

        job_id = result.get("job_id") or result.get("id")
        if not job_id:
            raise ValueError(f"Unexpected WhisperX response: {result}")

        logger.info(f"WhisperX job submitted: {job_id} — polling every {poll_interval}s")

        # --- Poll for result ---
        while True:
            time.sleep(poll_interval)
            status_resp = httpx.get(
                f"{self.api_url}/transcribe/{job_id}",
                headers=self.headers,
                timeout=30,
            )
            status_resp.raise_for_status()
            status = status_resp.json()

            state = status.get("status", "").lower()
            logger.info(f"  Job status: {state}")

            if state in ("done", "completed", "finished", "success"):
                srt_content = status.get("srt") or status.get("result") or status.get("output")
                if not srt_content:
                    raise ValueError(f"Job complete but no SRT in response: {status}")
                logger.info("Transcription complete!")
                return srt_content

            elif state in ("failed", "error"):
                raise RuntimeError(f"WhisperX transcription failed: {status}")

            # Still processing — keep polling
            progress = status.get("progress", "")
            if progress:
                logger.info(f"  Progress: {progress}")

    def save_srt(self, srt_content: str, output_path: str) -> Path:
        """Save SRT string to file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(srt_content, encoding="utf-8")
        logger.info(f"SRT saved: {out}")
        return out
