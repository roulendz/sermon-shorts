"""
tui/screens/segment_review_screen.py

Runs Manus AI segment selection on mount, then displays all selected
segments in a scrollable list with viral scores. The user reads through
them and clicks "Cut Videos" to proceed to the cutting screen.
"""

from __future__ import annotations

import os

from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Rule, Static
from textual import work

from models.pipeline_state import PipelineState
from models.video_segment import VideoSegment
from tui.widgets.pipeline_log import PipelineLog


class SegmentCard(Static):
    """Displays a single VideoSegment including transcript text and viral scores."""

    DEFAULT_CSS = """
    SegmentCard {
        border: solid $primary-darken-2;
        padding: 1 2;
        margin-bottom: 1;
        height: auto;
    }
    .segment-card-header {
        text-style: bold;
        color: $primary;
    }
    .segment-card-transcript {
        margin-bottom: 1;
    }
    .segment-card-viral-score {
        text-style: bold;
        color: $success;
    }
    .segment-card-hook-overlay {
        color: $warning;
    }
    .segment-card-reason {
        color: $text-muted;
        text-style: italic;
        margin-top: 1;
    }
    """

    def __init__(self, segment: VideoSegment) -> None:
        super().__init__()
        self._segment = segment

    def compose(self) -> ComposeResult:
        yield Label(
            f"Segment #{self._segment.index}: {self._segment.suggested_title}\n"
            f"{self._segment.ffmpeg_start_timestamp}  ->  {self._segment.ffmpeg_end_timestamp}"
            f"  ({self._segment.duration_seconds:.0f} sec)",
            classes="segment-card-header",
        )
        yield Label(
            f'"{self._segment.transcript_text}"',
            classes="segment-card-transcript",
        )
        yield Label(
            f"Viral score: {self._segment.overall_viral_score:.1f} / 10  "
            f"  Hook: {self._segment.hook_power_score}  "
            f"  Emotion: {self._segment.emotional_impact_score}  "
            f"  Discussion: {self._segment.discussion_potential_score}  "
            f"  Share: {self._segment.shareability_score}",
            classes="segment-card-viral-score",
        )
        if self._segment.viral_hook_text_overlay:
            yield Label(
                f'Opening overlay: "{self._segment.viral_hook_text_overlay}"',
                classes="segment-card-hook-overlay",
            )
        yield Label(
            f"Why: {self._segment.selection_reason}",
            classes="segment-card-reason",
        )


class SegmentReviewScreen(Screen):
    """
    Screen that runs Manus AI segment selection and then shows all
    segments for review before proceeding to video cutting.
    """

    BINDINGS = [("q", "quit", "Quit")]

    DEFAULT_CSS = """
    SegmentReviewScreen {
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
    #selection-log {
        margin-bottom: 1;
    }
    #segments-scroll-container {
        height: 1fr;
        border: none;
    }
    #cut-videos-button {
        margin-top: 1;
        width: 100%;
    }
    """

    def __init__(self, pipeline_state: PipelineState) -> None:
        super().__init__()
        self._pipeline_state = pipeline_state

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Label("Step 3 — Segment Review", id="screen-title")
            yield Label(
                "Manus AI is selecting the best segments from your sermon...",
                id="screen-subtitle",
            )
            yield PipelineLog(id="selection-log")
            yield Rule()
            yield ScrollableContainer(id="segments-scroll-container")
            yield Button(
                "Cut Videos →",
                id="cut-videos-button",
                variant="primary",
                disabled=True,
            )
        yield Footer()

    def on_mount(self) -> None:
        self._log_widget = self.query_one("#selection-log", PipelineLog)
        if self._pipeline_state.auto_segment_selection and self._pipeline_state.selected_segments:
            log = self._log_widget
            log.write_info("Auto segment selection: using existing segments")
            self._display_segments(self._pipeline_state.selected_segments)
            if self._pipeline_state.auto_segment_review:
                self._proceed_to_cutting_screen()
        else:
            self._run_segment_selection()

    @work(thread=True)
    def _run_segment_selection(self) -> None:
        from api.manus_client import ManusClient
        from pipeline.segment_selector import (
            build_segment_selection_prompt,
            parse_segments_from_manus_response,
        )
        from pipeline.subtitle_parser import load_subtitle_file

        log = self._log_widget

        manus_api_key = os.getenv("MANUS_API_KEY", "")
        if not manus_api_key:
            self.app.call_from_thread(log.write_error, "MANUS_API_KEY not set in .env")
            return

        self.app.call_from_thread(log.write_step_header, "Manus AI Segment Selection")
        self.app.call_from_thread(
            log.write_info,
            f"Loading subtitles from: {self._pipeline_state.subtitle_file_path.name}",
        )

        try:
            all_subtitles = load_subtitle_file(self._pipeline_state.subtitle_file_path)
            self.app.call_from_thread(
                log.write_info,
                f"Loaded {len(all_subtitles)} subtitle entries",
            )

            srt_content = "\n".join(
                f"{subtitle.index}\n{subtitle.start} --> {subtitle.end}\n{subtitle.content}\n"
                for subtitle in all_subtitles
            )

            prompt = build_segment_selection_prompt(srt_content)

            def on_progress(msg: str) -> None:
                self.app.call_from_thread(log.write_info, msg)

            manus_project_id = os.getenv("MANUS_PROJECT_ID", "")
            audio_filename = self._pipeline_state.audio_file_path.stem if self._pipeline_state.audio_file_path else None

            client = ManusClient(api_key=manus_api_key)
            manus_response = client.submit_prompt_and_wait_for_response(
                prompt,
                on_progress=on_progress,
                project_id=manus_project_id or None,
                task_title=audio_filename,
            )

            segments = parse_segments_from_manus_response(manus_response, all_subtitles)
            self._pipeline_state.selected_segments = segments

            self.app.call_from_thread(
                log.write_success,
                f"Manus selected {len(segments)} segments",
            )
            self.app.call_from_thread(self._display_segments, segments)

            if self._pipeline_state.auto_segment_review:
                self.app.call_from_thread(
                    log.write_info,
                    "Auto review enabled — proceeding to cutting automatically",
                )
                self.app.call_from_thread(self._proceed_to_cutting_screen)

        except Exception as error:
            self.app.call_from_thread(log.write_error, str(error))

    def _display_segments(self, segments: list[VideoSegment]) -> None:
        scroll_container = self.query_one("#segments-scroll-container", ScrollableContainer)

        for segment in segments:
            scroll_container.mount(SegmentCard(segment))

        self.query_one("#screen-subtitle", Label).update(
            f"{len(segments)} segments selected — review below, then click Cut Videos"
        )
        self.query_one("#cut-videos-button", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cut-videos-button":
            self._proceed_to_cutting_screen()

    def _proceed_to_cutting_screen(self) -> None:
        from tui.screens.cutting_screen import CuttingScreen
        self.app.push_screen(CuttingScreen(self._pipeline_state))

    def action_quit(self) -> None:
        self.app.exit()
