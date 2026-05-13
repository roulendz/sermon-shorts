"""
tui/screens/transcription_screen.py

Submits audio files to WhisperX for word-level transcription, stores full
JSON responses with versioning, and builds word-level SRT files.

Supports both LV and RU audio tracks simultaneously. If existing WhisperX
data is found, offers to reuse it or re-transcribe.

Also retains Transkriptor option for future Phase 2 word replacement.
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
    """Screen for transcribing audio via WhisperX (word-level) or Transkriptor."""

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
    #existing-data-notice {
        color: $success;
        text-style: bold;
        margin-bottom: 1;
    }
    #button-row {
        height: 3;
        margin-top: 1;
    }
    """

    def __init__(self, pipeline_state: PipelineState) -> None:
        super().__init__()
        self._pipeline_state = pipeline_state
        self._latest_version_number: int | None = None
        self._transkriptor_reuse_mode: bool = False
        self._pending_transkriptor_existing_path: Path | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Label("Step 2 -- Transcription", id="screen-title")
            yield Label(
                "Choose transcription service and start...",
                id="screen-subtitle",
            )
            yield Rule()
            yield Label("", id="existing-data-notice")
            yield PipelineLog(id="transcription-log")
            with Horizontal(id="button-row"):
                yield Button(
                    "Use Existing",
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
                yield Button(
                    "WhisperX Word-Level (LV+RU)",
                    id="whisperx-wordlevel-button",
                    variant="primary",
                )
                yield Button(
                    "Transkriptor (slow)",
                    id="transkriptor-button",
                    variant="default",
                )
                yield Button(
                    "Proceed to Segment Selection ->",
                    id="proceed-button",
                    variant="primary",
                    disabled=True,
                )
        yield Footer()

    def on_mount(self) -> None:
        self._log_widget = self.query_one("#transcription-log", PipelineLog)
        self._check_for_existing_whisperx_data()

    def _check_for_existing_whisperx_data(self) -> None:
        from pipeline.whisperx_response_storage import find_latest_version_number

        latest_version = find_latest_version_number(self._pipeline_state.video_file_path)

        if latest_version is not None:
            self._latest_version_number = latest_version
            notice = self.query_one("#existing-data-notice", Label)
            notice.update(f"Existing WhisperX data found: v{latest_version}")
            self.query_one("#use-existing-button", Button).disabled = False
            self.query_one("#retranscribe-button", Button).disabled = False
            self._log_widget.write_info(
                f"Found existing WhisperX data (v{latest_version}). "
                "Choose: Use Existing, or pick a service to re-transcribe."
            )
            self._hide_service_buttons()
        else:
            self._hide_choice_buttons()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "use-existing-button":
            self._disable_all_buttons()
            if self._transkriptor_reuse_mode:
                self._use_existing_transkriptor_transcription()
            else:
                self._load_existing_whisperx_data()
        elif event.button.id == "retranscribe-button":
            if self._transkriptor_reuse_mode:
                self._transkriptor_reuse_mode = False
                self._disable_all_buttons()
                self._run_transkriptor_transcription()
            else:
                self._hide_choice_buttons()
                self._show_service_buttons()
        elif event.button.id == "whisperx-wordlevel-button":
            self._disable_all_buttons()
            self._transcribe_all_audio_tracks()
        elif event.button.id == "transkriptor-button":
            self.on_button_pressed_transkriptor()
        elif event.button.id == "proceed-button":
            self._proceed_to_segment_selection_screen()

    @work(thread=True)
    def _transcribe_all_audio_tracks(self) -> None:
        from api.whisperx_client import WhisperXClient
        from pipeline.audio_compressor import prepare_audio_with_offset
        from pipeline.language_detect import language_name
        from pipeline.whisperx_response_storage import (
            build_next_version_directory_path,
            save_whisperx_response,
        )
        from pipeline.word_level_srt_builder import (
            build_cleaned_speech_regions,
            build_word_level_subtitles,
            extract_words_from_response,
            map_words_to_speech_regions,
            merge_bilingual_word_level_srt,
            save_word_level_srt,
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

        self.app.call_from_thread(
            log.write_step_header,
            "WhisperX Word-Level Transcription (all tracks)",
        )

        discovered_tracks = self._pipeline_state.discovered_audio_tracks
        self.app.call_from_thread(
            log.write_info,
            f"Processing {len(discovered_tracks)} audio track(s): "
            + ", ".join(
                f"{language_name(language_code)} ({audio_path.name})"
                for language_code, audio_path in discovered_tracks.items()
            ),
        )

        def on_progress(message: str) -> None:
            self.app.call_from_thread(log.write_info, message)

        try:
            client = WhisperXClient(
                api_base_url=whisperx_api_url,
                api_key=whisperx_api_key,
            )

            version_directory = build_next_version_directory_path(
                self._pipeline_state.video_file_path
            )
            self.app.call_from_thread(
                log.write_info,
                f"Storing results in: {version_directory.name}/",
            )

            primary_subtitle_file_path = None
            responses_by_language: dict[str, dict] = {}

            for language_code, audio_file_path in discovered_tracks.items():
                self.app.call_from_thread(
                    log.write_step_header,
                    f"Transcribing {language_name(language_code)} ({audio_file_path.name})",
                )

                # 1. Prepare audio with baked-in offset
                prepared_audio_path = prepare_audio_with_offset(
                    audio_file_path,
                    on_progress=on_progress,
                )

                # 2. Transcribe via WhisperX (raw JSON with word-level data)
                raw_response = client.transcribe_audio_file_raw(
                    audio_file_path=prepared_audio_path,
                    language_code=language_code,
                    on_progress=on_progress,
                )

                # 3. Store versioned JSON
                json_path = save_whisperx_response(
                    video_file_path=self._pipeline_state.video_file_path,
                    audio_file_stem=audio_file_path.stem,
                    response_data=raw_response,
                    version_directory=version_directory,
                )
                self.app.call_from_thread(
                    log.write_info,
                    f"Response saved: {json_path.name}",
                )

                # 4. Build word-level SRT (single-language)
                subtitles = build_word_level_subtitles(raw_response)
                responses_by_language[language_code] = raw_response

                srt_output_path = version_directory / f"{audio_file_path.stem}_wordlevel.srt"
                save_word_level_srt(subtitles, srt_output_path)
                self.app.call_from_thread(
                    log.write_info,
                    f"Word-level SRT: {srt_output_path.name} ({len(subtitles)} entries)",
                )

                # 5. Also save to transcriptions/ for downstream pipeline compatibility
                transcriptions_directory = (
                    self._pipeline_state.video_file_path.parent / "transcriptions"
                )
                compatibility_srt_path = (
                    transcriptions_directory / f"{audio_file_path.stem}_wordlevel.srt"
                )
                save_word_level_srt(subtitles, compatibility_srt_path)

                # Use LV as the primary SRT for downstream pipeline
                if language_code == "lv" or primary_subtitle_file_path is None:
                    primary_subtitle_file_path = compatibility_srt_path

                self.app.call_from_thread(
                    log.write_success,
                    f"{language_name(language_code)} track complete",
                )

            # 6. Merge bilingual SRT if both LV and RU are available
            if "lv" in responses_by_language and "ru" in responses_by_language:
                self.app.call_from_thread(
                    log.write_step_header,
                    "Building bilingual SRT (speech regions + word mapping)",
                )

                from pipeline.audio_compressor import DEFAULT_AUDIO_OFFSET_SECONDS

                ru_speech, lv_speech = build_cleaned_speech_regions(
                    primary_audio_path=discovered_tracks["ru"],
                    secondary_audio_path=discovered_tracks["lv"],
                    offset_seconds=DEFAULT_AUDIO_OFFSET_SECONDS,
                    on_progress=on_progress,
                )

                ru_words = extract_words_from_response(responses_by_language["ru"])
                lv_words = extract_words_from_response(responses_by_language["lv"])

                ru_subtitles = map_words_to_speech_regions(ru_words, ru_speech, "RU")
                lv_subtitles = map_words_to_speech_regions(lv_words, lv_speech, "LV")

                self.app.call_from_thread(
                    log.write_info,
                    f"Mapped: RU={len(ru_subtitles)} subs, LV={len(lv_subtitles)} subs",
                )

                merged_subtitles = merge_bilingual_word_level_srt(
                    ru_subtitles, lv_subtitles,
                )

                video_stem = self._pipeline_state.video_file_path.stem
                merged_srt_path = version_directory / f"{video_stem}_bilingual_wordlevel.srt"
                save_word_level_srt(merged_subtitles, merged_srt_path)

                transcriptions_directory = (
                    self._pipeline_state.video_file_path.parent / "transcriptions"
                )
                merged_compatibility_path = (
                    transcriptions_directory / f"{video_stem}_bilingual_wordlevel.srt"
                )
                save_word_level_srt(merged_subtitles, merged_compatibility_path)

                self.app.call_from_thread(
                    log.write_success,
                    f"Bilingual SRT: {merged_srt_path.name} ({len(merged_subtitles)} entries)",
                )
                primary_subtitle_file_path = merged_compatibility_path

            self._pipeline_state.subtitle_file_path = primary_subtitle_file_path
            self.app.call_from_thread(
                log.write_success,
                f"All tracks transcribed. Primary SRT: {primary_subtitle_file_path.name}",
            )
            self.app.call_from_thread(self._enable_proceed_button)

        except Exception as error:
            self.app.call_from_thread(
                log.write_error,
                f"{type(error).__name__}: {error}",
            )

    @work(thread=True)
    def _load_existing_whisperx_data(self) -> None:
        from pipeline.whisperx_response_storage import (
            build_version_directory_path,
            list_stored_audio_stems_for_version,
            load_whisperx_response,
        )
        from pipeline.word_level_srt_builder import (
            build_cleaned_speech_regions,
            build_word_level_subtitles,
            extract_words_from_response,
            map_words_to_speech_regions,
            merge_bilingual_word_level_srt,
            save_word_level_srt,
        )

        log = self._log_widget
        version_number = self._latest_version_number

        self.app.call_from_thread(
            log.write_step_header,
            f"Loading existing WhisperX data (v{version_number})",
        )

        try:
            version_directory = build_version_directory_path(
                self._pipeline_state.video_file_path,
                version_number,
            )

            stored_stems = list_stored_audio_stems_for_version(
                self._pipeline_state.video_file_path,
                version_number,
            )

            if not stored_stems:
                self.app.call_from_thread(
                    log.write_error,
                    f"No stored responses found in v{version_number}/",
                )
                return

            self.app.call_from_thread(
                log.write_info,
                f"Found {len(stored_stems)} stored response(s): {', '.join(stored_stems)}",
            )

            primary_subtitle_file_path = None
            responses_by_language: dict[str, dict] = {}

            for audio_stem in stored_stems:
                response_data = load_whisperx_response(
                    self._pipeline_state.video_file_path,
                    audio_stem,
                    version_number,
                )

                # Build single-language word-level SRT
                subtitles = build_word_level_subtitles(response_data)
                srt_path = version_directory / f"{audio_stem}_wordlevel.srt"
                save_word_level_srt(subtitles, srt_path)
                self.app.call_from_thread(
                    log.write_info,
                    f"Word-level SRT: {srt_path.name} ({len(subtitles)} entries)",
                )

                # Track responses by language for bilingual merge
                if "_A03" in audio_stem:
                    responses_by_language["lv"] = response_data
                elif "_A04" in audio_stem:
                    responses_by_language["ru"] = response_data

                # Copy to transcriptions/ for downstream compatibility
                transcriptions_directory = (
                    self._pipeline_state.video_file_path.parent / "transcriptions"
                )
                compatibility_srt_path = (
                    transcriptions_directory / f"{audio_stem}_wordlevel.srt"
                )
                save_word_level_srt(subtitles, compatibility_srt_path)

                # LV (_A03) is the primary track for downstream
                if "_A03" in audio_stem or primary_subtitle_file_path is None:
                    primary_subtitle_file_path = compatibility_srt_path

            # Merge bilingual SRT if both LV and RU are available
            if "lv" in responses_by_language and "ru" in responses_by_language:
                self.app.call_from_thread(
                    log.write_step_header,
                    "Building bilingual SRT (speech regions + word mapping)",
                )

                discovered_tracks = self._pipeline_state.discovered_audio_tracks

                def on_progress(message: str) -> None:
                    self.app.call_from_thread(log.write_info, message)

                from pipeline.audio_compressor import DEFAULT_AUDIO_OFFSET_SECONDS

                ru_speech, lv_speech = build_cleaned_speech_regions(
                    primary_audio_path=discovered_tracks["ru"],
                    secondary_audio_path=discovered_tracks["lv"],
                    offset_seconds=DEFAULT_AUDIO_OFFSET_SECONDS,
                    on_progress=on_progress,
                )

                ru_words = extract_words_from_response(responses_by_language["ru"])
                lv_words = extract_words_from_response(responses_by_language["lv"])

                ru_subtitles = map_words_to_speech_regions(ru_words, ru_speech, "RU")
                lv_subtitles = map_words_to_speech_regions(lv_words, lv_speech, "LV")

                self.app.call_from_thread(
                    log.write_info,
                    f"Mapped: RU={len(ru_subtitles)} subs, LV={len(lv_subtitles)} subs",
                )

                merged_subtitles = merge_bilingual_word_level_srt(
                    ru_subtitles, lv_subtitles,
                )

                video_stem = self._pipeline_state.video_file_path.stem
                merged_srt_path = version_directory / f"{video_stem}_bilingual_wordlevel.srt"
                save_word_level_srt(merged_subtitles, merged_srt_path)

                transcriptions_directory = (
                    self._pipeline_state.video_file_path.parent / "transcriptions"
                )
                merged_compatibility_path = (
                    transcriptions_directory / f"{video_stem}_bilingual_wordlevel.srt"
                )
                save_word_level_srt(merged_subtitles, merged_compatibility_path)

                self.app.call_from_thread(
                    log.write_success,
                    f"Bilingual SRT: {merged_srt_path.name} ({len(merged_subtitles)} entries)",
                )
                primary_subtitle_file_path = merged_compatibility_path

            self._pipeline_state.subtitle_file_path = primary_subtitle_file_path
            self._pipeline_state.whisperx_version = version_number
            self.app.call_from_thread(
                log.write_success,
                f"Loaded v{version_number} data. Primary SRT: {primary_subtitle_file_path.name}",
            )
            self.app.call_from_thread(self._enable_proceed_button)

        except Exception as error:
            self.app.call_from_thread(
                log.write_error,
                f"{type(error).__name__}: {error}",
            )

    def _delete_intermediate_audio_file(
        self,
        intermediate_path: Path,
        original_path: Path,
        on_progress,
    ) -> None:
        """Delete compressed audio file after successful upload. Never deletes originals."""
        if intermediate_path == original_path:
            return
        if intermediate_path.exists():
            intermediate_size_megabytes = intermediate_path.stat().st_size / (1024 * 1024)
            intermediate_path.unlink()
            message = f"Cleaned up intermediate file: {intermediate_path.name} ({intermediate_size_megabytes:.0f} MB freed)"
            self.app.call_from_thread(self._log_widget.write_info, message)

    def _find_existing_transkriptor_transcription(self) -> Path | None:
        """Find most recent Transkriptor SRT in transcriptions/ folder."""
        from pipeline.transcription_runner import SERVICE_TRANSKRIPTOR

        transcriptions_directory = (
            self._pipeline_state.video_file_path.parent / "transcriptions"
        )
        if not transcriptions_directory.exists():
            return None

        transkriptor_files = sorted(
            transcriptions_directory.glob(f"*_{SERVICE_TRANSKRIPTOR}_*.srt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return transkriptor_files[0] if transkriptor_files else None

    def on_button_pressed_transkriptor(self) -> None:
        existing_transcription_path = self._find_existing_transkriptor_transcription()

        if existing_transcription_path is not None:
            self._pending_transkriptor_existing_path = existing_transcription_path
            self._show_transkriptor_reuse_choice(existing_transcription_path)
        else:
            self._disable_all_buttons()
            self._run_transkriptor_transcription()

    def _show_transkriptor_reuse_choice(self, existing_path: Path) -> None:
        log = self._log_widget
        log.write_info(
            f"Existing Transkriptor transcription found: {existing_path.name}"
        )
        self._hide_service_buttons()
        self._hide_choice_buttons()

        use_existing_button = self.query_one("#use-existing-button", Button)
        use_existing_button.label = "Use Existing Transkriptor"
        use_existing_button.display = True
        use_existing_button.disabled = False

        retranscribe_button = self.query_one("#retranscribe-button", Button)
        retranscribe_button.label = "Re-transcribe (Transkriptor)"
        retranscribe_button.display = True
        retranscribe_button.disabled = False

        self._transkriptor_reuse_mode = True

    def _use_existing_transkriptor_transcription(self) -> None:
        log = self._log_widget
        existing_path = self._pending_transkriptor_existing_path
        self._pipeline_state.subtitle_file_path = existing_path
        log.write_success(f"Using existing transcription: {existing_path.name}")
        self._enable_proceed_button()

    @work(thread=True)
    def _run_transkriptor_transcription(self) -> None:
        from api.transkriptor_client import TranskriptorClient
        from pipeline.language_detect import (
            detect_language_from_filename,
            language_name,
            to_transkriptor_locale,
        )

        log = self._log_widget

        transkriptor_api_key = os.getenv("TRANSKRIPTOR_API_KEY", "")

        if not transkriptor_api_key:
            self.app.call_from_thread(
                log.write_error,
                "TRANSKRIPTOR_API_KEY not set in .env",
            )
            return

        discovered_tracks = self._pipeline_state.discovered_audio_tracks
        if not discovered_tracks:
            self.app.call_from_thread(log.write_error, "No audio tracks available")
            return

        audio_file_path = discovered_tracks.get("lv") or next(iter(discovered_tracks.values()))
        language_code = detect_language_from_filename(audio_file_path)
        transkriptor_locale = to_transkriptor_locale(language_code)

        self.app.call_from_thread(log.write_step_header, "Transkriptor Transcription")
        self.app.call_from_thread(
            log.write_info,
            f"Audio file: {audio_file_path.name}",
        )
        self.app.call_from_thread(
            log.write_info,
            f"Language: {language_name(language_code)} ({transkriptor_locale})",
        )

        def on_progress(message: str) -> None:
            self.app.call_from_thread(log.write_info, message)

        try:
            from pipeline.audio_compressor import compress_audio_for_transkriptor
            from pipeline.transcription_runner import (
                transcribe_with_transkriptor_and_save_aligned_subtitles,
                build_transcription_output_path,
                SERVICE_TRANSKRIPTOR,
            )

            upload_path = compress_audio_for_transkriptor(
                audio_file_path,
                on_progress=on_progress,
            )

            video_file_stem = self._pipeline_state.video_file_path.stem
            client = TranskriptorClient(api_key=transkriptor_api_key)
            output_path = build_transcription_output_path(
                video_file_path=self._pipeline_state.video_file_path,
                audio_file_path=audio_file_path,
                service_code=SERVICE_TRANSKRIPTOR,
            )

            subtitle_file_path = transcribe_with_transkriptor_and_save_aligned_subtitles(
                audio_file_path=upload_path,
                output_subtitle_file_path=output_path,
                transkriptor_client=client,
                audio_to_video_offset_seconds=0.0,
                language_locale=transkriptor_locale,
                display_file_name=video_file_stem,
                on_progress=on_progress,
            )

            self._delete_intermediate_audio_file(upload_path, audio_file_path, on_progress)

            self._pipeline_state.subtitle_file_path = subtitle_file_path
            self.app.call_from_thread(
                log.write_success,
                f"Subtitles saved: {subtitle_file_path}",
            )
            self.app.call_from_thread(self._enable_proceed_button)

        except Exception as error:
            self.app.call_from_thread(
                log.write_error,
                f"{type(error).__name__}: {error}",
            )

    def _hide_service_buttons(self) -> None:
        self.query_one("#whisperx-wordlevel-button", Button).display = False
        self.query_one("#transkriptor-button", Button).display = False

    def _show_service_buttons(self) -> None:
        self.query_one("#whisperx-wordlevel-button", Button).display = True
        self.query_one("#transkriptor-button", Button).display = True

    def _hide_choice_buttons(self) -> None:
        self.query_one("#use-existing-button", Button).display = False
        self.query_one("#retranscribe-button", Button).display = False

    def _disable_all_buttons(self) -> None:
        self.query_one("#use-existing-button", Button).disabled = True
        self.query_one("#retranscribe-button", Button).disabled = True
        self.query_one("#whisperx-wordlevel-button", Button).disabled = True
        self.query_one("#transkriptor-button", Button).disabled = True

    def _enable_proceed_button(self) -> None:
        proceed_button = self.query_one("#proceed-button", Button)
        proceed_button.disabled = False

    def _proceed_to_segment_selection_screen(self) -> None:
        from tui.screens.segment_review_screen import SegmentReviewScreen
        self.app.push_screen(SegmentReviewScreen(self._pipeline_state))

    def action_quit(self) -> None:
        self.app.exit()
