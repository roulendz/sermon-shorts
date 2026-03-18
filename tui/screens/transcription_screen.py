"""
tui/screens/transcription_screen.py

Submits the audio file to WhisperX or Transkriptor for transcription, applies
the audio-to-video offset to align subtitle timestamps, and saves the SRT file.
If an SRT already exists for the audio file, offers to reuse it.
"""

from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Rule
from textual import work

from models.pipeline_state import PipelineState
from tui.widgets.pipeline_log import PipelineLog


class TranscriptionScreen(Screen):
    """Screen for transcribing audio via WhisperX or Transkriptor and saving aligned SRT."""

    BINDINGS = [("q", "quit", "Quit")]

    DEFAULT_CSS = """
    TranscriptionScreen {
        padding: 1 2;
    }
    #screen-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 0;
    }
    #screen-subtitle {
        color: $text-muted;
        margin-bottom: 1;
    }
    #existing-srt-notice {
        color: $success;
        text-style: bold;
        margin-bottom: 1;
    }
    #choice-row {
        height: 3;
        margin-bottom: 1;
    }
    #service-row {
        height: 3;
        margin-bottom: 1;
    }
    #proceed-button {
        margin-top: 2;
        width: 100%;
    }
    """

    def __init__(self, pipeline_state: PipelineState) -> None:
        super().__init__()
        self._pipeline_state = pipeline_state

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Label("Step 2 -- Transcription", id="screen-title")
            yield Label(
                "Choose transcription service and start...",
                id="screen-subtitle",
            )
            yield Rule()
            yield Label("", id="existing-srt-notice")
            with Horizontal(id="choice-row"):
                yield Button(
                    "Use Existing SRT",
                    id="use-existing-button",
                    variant="success",
                    disabled=True,
                )
                yield Button(
                    "Re-transcribe",
                    id="retranscribe-button",
                    variant="warning",
                    disabled=True,
                )
            with Horizontal(id="service-row"):
                yield Button(
                    "WhisperX (fast)",
                    id="whisperx-button",
                    variant="primary",
                )
                yield Button(
                    "Transkriptor (slow)",
                    id="transkriptor-button",
                    variant="default",
                )
            yield PipelineLog(id="transcription-log")
            yield Button(
                "Proceed to Segment Selection ->",
                id="proceed-button",
                variant="primary",
                disabled=True,
            )
        yield Footer()

    def on_mount(self) -> None:
        self._log_widget = self.query_one("#transcription-log", PipelineLog)
        existing_srt_path = self._find_existing_transcription()

        if existing_srt_path:
            notice = self.query_one("#existing-srt-notice", Label)
            notice.update(f"Existing SRT found: {existing_srt_path.name}")
            self.query_one("#use-existing-button", Button).disabled = False
            self.query_one("#retranscribe-button", Button).disabled = False
            self._log_widget.write_info(f"Found existing transcript: {existing_srt_path}")
            self._log_widget.write_info("Choose: Use Existing, or pick a service to re-transcribe")
            self._hide_service_buttons()
        else:
            self._hide_choice_buttons()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "use-existing-button":
            srt_path = self._find_existing_transcription()
            if srt_path:
                self._pipeline_state.subtitle_file_path = srt_path
                self._log_widget.write_success(f"Using existing: {srt_path.name}")
                self._enable_proceed_button()
        elif event.button.id == "retranscribe-button":
            self._hide_choice_buttons()
            self._show_service_buttons()
        elif event.button.id == "whisperx-button":
            self._disable_all_service_buttons()
            self._start_whisperx_transcription()
        elif event.button.id == "transkriptor-button":
            self._disable_all_service_buttons()
            self._start_transkriptor_transcription()
        elif event.button.id == "proceed-button":
            self._proceed_to_segment_selection_screen()

    @work(thread=True)
    def _start_whisperx_transcription(self) -> None:
        from api.whisperx_client import WhisperXClient
        from pipeline.transcription_runner import (
            transcribe_and_save_aligned_subtitles,
            build_transcription_output_path,
            SERVICE_WHISPERX,
        )

        log = self._log_widget

        whisperx_api_url = os.getenv("WHISPERX_API_URL", "")
        whisperx_api_key = os.getenv("WHISPERX_API_KEY", "")

        if not whisperx_api_url or not whisperx_api_key:
            self.app.call_from_thread(
                log.write_error,
                "WHISPERX_API_URL or WHISPERX_API_KEY not set in .env",
            )
            return

        from pipeline.language_detect import detect_language_from_filename, language_name

        language_code = detect_language_from_filename(self._pipeline_state.audio_file_path)

        self.app.call_from_thread(log.write_step_header, "WhisperX Transcription")
        self.app.call_from_thread(
            log.write_info,
            f"Audio file: {self._pipeline_state.audio_file_path.name}",
        )
        self.app.call_from_thread(
            log.write_info,
            f"Language: {language_name(language_code)} ({language_code})",
        )
        self.app.call_from_thread(
            log.write_info,
            f"Offset to apply: {self._pipeline_state.audio_to_video_offset_seconds:.3f}s",
        )

        def on_progress(message: str) -> None:
            self.app.call_from_thread(log.write_info, message)

        try:
            from pipeline.audio_compressor import compress_audio_if_needed
            upload_path = compress_audio_if_needed(
                self._pipeline_state.audio_file_path,
                on_progress=on_progress,
            )

            client = WhisperXClient(
                api_base_url=whisperx_api_url,
                api_key=whisperx_api_key,
            )
            output_path = build_transcription_output_path(
                video_file_path=self._pipeline_state.video_file_path,
                audio_file_path=self._pipeline_state.audio_file_path,
                service_code=SERVICE_WHISPERX,
            )

            subtitle_file_path = transcribe_and_save_aligned_subtitles(
                audio_file_path=upload_path,
                output_subtitle_file_path=output_path,
                whisperx_client=client,
                audio_to_video_offset_seconds=self._pipeline_state.audio_to_video_offset_seconds,
                language_code=language_code,
                on_progress=on_progress,
            )
            self._pipeline_state.subtitle_file_path = subtitle_file_path
            self.app.call_from_thread(log.write_success, f"Subtitles saved: {subtitle_file_path}")
            self.app.call_from_thread(self._enable_proceed_button)

        except Exception as error:
            self.app.call_from_thread(log.write_error, f"{type(error).__name__}: {error}")

    @work(thread=True)
    def _start_transkriptor_transcription(self) -> None:
        from api.transkriptor_client import TranskriptorClient
        from pipeline.transcription_runner import (
            transcribe_with_transkriptor_and_save_aligned_subtitles,
            build_transcription_output_path,
            SERVICE_TRANSKRIPTOR,
        )

        log = self._log_widget

        transkriptor_api_key = os.getenv("TRANSKRIPTOR_API_KEY", "")

        if not transkriptor_api_key:
            self.app.call_from_thread(
                log.write_error,
                "TRANSKRIPTOR_API_KEY not set in .env",
            )
            return

        from pipeline.language_detect import (
            detect_language_from_filename,
            language_name,
            to_transkriptor_locale,
        )

        language_code = detect_language_from_filename(self._pipeline_state.audio_file_path)
        transkriptor_locale = to_transkriptor_locale(language_code)

        self.app.call_from_thread(log.write_step_header, "Transkriptor Transcription")
        self.app.call_from_thread(
            log.write_info,
            f"Audio file: {self._pipeline_state.audio_file_path.name}",
        )
        self.app.call_from_thread(
            log.write_info,
            f"Language: {language_name(language_code)} ({transkriptor_locale})",
        )
        self.app.call_from_thread(
            log.write_info,
            f"Offset to apply: {self._pipeline_state.audio_to_video_offset_seconds:.3f}s",
        )

        def on_progress(message: str) -> None:
            self.app.call_from_thread(log.write_info, message)

        try:
            from pipeline.audio_compressor import compress_audio_if_needed
            upload_path = compress_audio_if_needed(
                self._pipeline_state.audio_file_path,
                on_progress=on_progress,
            )

            client = TranskriptorClient(api_key=transkriptor_api_key)
            output_path = build_transcription_output_path(
                video_file_path=self._pipeline_state.video_file_path,
                audio_file_path=self._pipeline_state.audio_file_path,
                service_code=SERVICE_TRANSKRIPTOR,
            )

            subtitle_file_path = transcribe_with_transkriptor_and_save_aligned_subtitles(
                audio_file_path=upload_path,
                output_subtitle_file_path=output_path,
                transkriptor_client=client,
                audio_to_video_offset_seconds=self._pipeline_state.audio_to_video_offset_seconds,
                language_locale=transkriptor_locale,
                on_progress=on_progress,
            )
            self._pipeline_state.subtitle_file_path = subtitle_file_path
            self.app.call_from_thread(log.write_success, f"Subtitles saved: {subtitle_file_path}")
            self.app.call_from_thread(self._enable_proceed_button)

        except Exception as error:
            self.app.call_from_thread(log.write_error, f"{type(error).__name__}: {error}")

    def _find_existing_transcription(self) -> Path | None:
        """Look for any existing SRT in the transcriptions folder for this audio."""
        if not self._pipeline_state.video_file_path:
            return None
        transcriptions_directory = self._pipeline_state.video_file_path.parent / "transcriptions"
        if not transcriptions_directory.exists():
            return None
        audio_stem = self._pipeline_state.audio_file_path.stem
        matching_files = sorted(
            transcriptions_directory.glob(f"{audio_stem}_*.srt"),
            reverse=True,
        )
        return matching_files[0] if matching_files else None

    def _hide_service_buttons(self) -> None:
        self.query_one("#whisperx-button", Button).display = False
        self.query_one("#transkriptor-button", Button).display = False

    def _show_service_buttons(self) -> None:
        self.query_one("#whisperx-button", Button).display = True
        self.query_one("#transkriptor-button", Button).display = True

    def _hide_choice_buttons(self) -> None:
        self.query_one("#use-existing-button", Button).display = False
        self.query_one("#retranscribe-button", Button).display = False

    def _disable_all_service_buttons(self) -> None:
        self.query_one("#whisperx-button", Button).disabled = True
        self.query_one("#transkriptor-button", Button).disabled = True

    def _enable_proceed_button(self) -> None:
        proceed_button = self.query_one("#proceed-button", Button)
        proceed_button.disabled = False

    def _proceed_to_segment_selection_screen(self) -> None:
        from tui.screens.segment_review_screen import SegmentReviewScreen
        self.app.push_screen(SegmentReviewScreen(self._pipeline_state))

    def action_quit(self) -> None:
        self.app.exit()
