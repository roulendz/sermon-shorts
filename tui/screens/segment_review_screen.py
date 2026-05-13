"""
tui/screens/segment_review_screen.py

Runs Manus AI segment selection on mount, then displays all selected
segments in a scrollable list with viral scores. The user reads through
them and clicks "Cut Videos" to proceed to the cutting screen.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, RadioButton, RadioSet, Rule, Static
from textual import work

from models.pipeline_state import PipelineState
from models.video_segment import VideoSegment
from tui.widgets.pipeline_log import PipelineLog

logger = logging.getLogger(__name__)


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
    #main-columns {
        height: auto;
        margin-bottom: 1;
    }
    #source-container {
        width: 1fr;
        height: auto;
        padding: 1;
        border: solid $primary-darken-2;
    }
    #source-label {
        text-style: bold;
        margin-bottom: 0;
    }
    #source-radio {
        height: auto;
        margin-top: 0;
    }
    .inline-field {
        height: 3;
        margin-top: 0;
    }
    .inline-field Label {
        width: auto;
        padding: 1 1 0 0;
    }
    .inline-field Input {
        width: 1fr;
    }
    #resume-task-button {
        width: 100%;
    }
    #action-container {
        width: 1fr;
        height: auto;
        padding: 1;
        margin-left: 1;
        border: solid $warning-darken-1;
    }
    #action-label {
        text-style: bold;
        color: $warning;
        margin-bottom: 0;
    }
    #use-selected-button {
        width: 100%;
        margin-top: 1;
    }
    #new-manus-button {
        width: 100%;
        margin-top: 0;
    }
    #selection-log {
        margin-top: 1;
        height: 7;
        max-height: 7;
    }
    #segments-scroll-container {
        height: auto;
        max-height: 50%;
    }
    #cut-videos-button {
        margin-top: 1;
        width: 100%;
    }
    """

    def __init__(self, pipeline_state: PipelineState) -> None:
        super().__init__()
        self._pipeline_state = pipeline_state
        self._source_entries: list[dict] = []
        self._selected_source_index: int = -1

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Label("Step 3 — Segment Review", id="screen-title")
            yield Label(
                "Select a Manus response or request new analysis",
                id="screen-subtitle",
            )
            yield Rule()
            with Horizontal(id="main-columns"):
                with Vertical(id="source-container"):
                    yield Label("Available Responses", id="source-label")
                    yield RadioSet(id="source-radio")
                    yield Rule()
                    with Horizontal(classes="inline-field"):
                        yield Label("URL/ID:")
                        yield Input(
                            placeholder="Paste Manus task URL or ID...",
                            id="task-url-input",
                        )
                    yield Button(
                        "Resume Task from URL",
                        id="resume-task-button",
                        variant="default",
                    )
                with Vertical(id="action-container"):
                    yield Label("Actions", id="action-label")
                    yield Button(
                        "Use Selected Response",
                        id="use-selected-button",
                        variant="success",
                        disabled=True,
                    )
                    yield Button(
                        "Request New Analysis",
                        id="new-manus-button",
                        variant="warning",
                    )
                    yield PipelineLog(id="selection-log")
                    yield Button(
                        "Cut Videos ->",
                        id="cut-videos-button",
                        variant="primary",
                        disabled=True,
                    )
            yield Rule()
            yield ScrollableContainer(id="segments-scroll-container")
        yield Footer()

    def on_mount(self) -> None:
        self._log_widget = self.query_one("#selection-log", PipelineLog)
        if self._pipeline_state.auto_segment_selection and self._pipeline_state.selected_segments:
            log = self._log_widget
            log.write_info("Auto segment selection: using existing segments")
            self._display_segments(self._pipeline_state.selected_segments)
            if self._pipeline_state.auto_segment_review:
                self._proceed_to_cutting_screen()
            return

        self._populate_source_list()

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
            video_filename = self._pipeline_state.video_file_path.stem if self._pipeline_state.video_file_path else None

            def on_task_created(task_id: str) -> None:
                self._pipeline_state.manus_task_id = task_id
                self._register_task(task_id, status="running")
                self.app.call_from_thread(
                    log.write_info, f"Task registered for resume: {task_id}"
                )

            client = ManusClient(api_key=manus_api_key)
            manus_response = client.submit_prompt_and_wait_for_response(
                prompt,
                on_progress=on_progress,
                on_task_created=on_task_created,
                project_id=manus_project_id or None,
                task_title=video_filename,
            )

            # Save Manus response to JSON for reuse
            self._save_manus_response(manus_response)
            if self._pipeline_state.manus_task_id:
                self._update_task_status(
                    self._pipeline_state.manus_task_id, "completed"
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
                    "Auto review enabled -- proceeding to cutting automatically",
                )
                self.app.call_from_thread(self._proceed_to_cutting_screen)

        except Exception as error:
            self.app.call_from_thread(log.write_error, str(error))

    def _display_segments(self, segments: list[VideoSegment]) -> None:
        scroll_container = self.query_one("#segments-scroll-container", ScrollableContainer)

        for segment in segments:
            scroll_container.mount(SegmentCard(segment))

        self.query_one("#screen-subtitle", Label).update(
            f"{len(segments)} segments selected -- review below, then click Cut Videos"
        )
        self.query_one("#cut-videos-button", Button).disabled = False

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "source-radio":
            self._selected_source_index = event.index
            self.query_one("#use-selected-button", Button).disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "use-selected-button":
            self._use_selected_source()
        elif event.button.id == "new-manus-button":
            self._disable_source_controls()
            self._run_segment_selection()
        elif event.button.id == "resume-task-button":
            task_url_or_id = self.query_one("#task-url-input", Input).value.strip()
            if not task_url_or_id:
                self._log_widget.write_error("Enter a Manus task URL or ID first")
                return
            self._disable_source_controls()
            self._resume_manus_task(task_url_or_id)
        elif event.button.id == "cut-videos-button":
            self._proceed_to_cutting_screen()

    def _populate_source_list(self) -> None:
        """Find all cached Manus responses and task history, populate radio buttons."""
        radio_set = self.query_one("#source-radio", RadioSet)
        self._source_entries = []

        cached_files = self._find_all_manus_responses()
        for response_path in cached_files:
            size_kilobytes = response_path.stat().st_size // 1024
            label = f"{response_path.stem} ({size_kilobytes}KB)"
            self._source_entries.append({"type": "file", "path": response_path})
            radio_set.mount(RadioButton(label))

        tasks = self._load_task_registry()
        for task in reversed(tasks):
            task_id = task.get("task_id", "?")
            status = task.get("status", "unknown")
            submitted = task.get("submitted_at", "")[:16]
            has_cached_file = any(
                entry.get("type") == "file" for entry in self._source_entries
            )
            if status == "completed" and has_cached_file:
                continue
            label = f"[{status}] {task_id[:16]}... {submitted}"
            self._source_entries.append({"type": "task", "task_id": task_id})
            radio_set.mount(RadioButton(label))

        if not self._source_entries:
            self._log_widget.write_info("No existing Manus responses found")

    def _use_selected_source(self) -> None:
        """Load segments from whatever source the user selected in the radio."""
        if self._selected_source_index < 0 or self._selected_source_index >= len(self._source_entries):
            self._log_widget.write_error("Select a response first")
            return
        entry = self._source_entries[self._selected_source_index]
        self._disable_source_controls()
        if entry["type"] == "file":
            self._load_manus_response_from_file(entry["path"])
        elif entry["type"] == "task":
            self._resume_manus_task(entry["task_id"])

    def _disable_source_controls(self) -> None:
        """Disable all source selection controls once an action is chosen."""
        self.query_one("#use-selected-button", Button).disabled = True
        self.query_one("#new-manus-button", Button).disabled = True
        self.query_one("#resume-task-button", Button).disabled = True

    def _find_all_manus_responses(self) -> list[Path]:
        """Find all saved Manus response JSON files next to the video file."""
        if not self._pipeline_state.video_file_path:
            return []
        video_directory = self._pipeline_state.video_file_path.parent
        return sorted(
            video_directory.glob("manus_response_*.json"),
            reverse=True,
        )

    def _save_manus_response(self, response_text: str) -> None:
        """Save Manus raw response to JSON file for reuse."""
        if not self._pipeline_state.video_file_path:
            return
        video_directory = self._pipeline_state.video_file_path.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = video_directory / f"manus_response_{timestamp}.json"
        output_path.write_text(response_text, encoding="utf-8", errors="replace")
        logger.info(f"Manus response saved: {output_path}")

    @work(thread=True)
    def _load_manus_response_from_file(self, response_path: Path) -> None:
        """Load segments from a previously saved Manus response file."""
        from pipeline.segment_selector import parse_segments_from_manus_response
        from pipeline.subtitle_parser import load_subtitle_file

        log = self._log_widget

        try:
            self.app.call_from_thread(log.write_step_header, "Loading Cached Response")
            self.app.call_from_thread(log.write_info, f"File: {response_path.name}")
            try:
                response_text = response_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                response_text = response_path.read_text(encoding="utf-8", errors="replace")

            self.app.call_from_thread(
                log.write_info,
                f"Response size: {len(response_text)} characters",
            )

            all_subtitles = load_subtitle_file(self._pipeline_state.subtitle_file_path)
            self.app.call_from_thread(
                log.write_info,
                f"Loaded {len(all_subtitles)} subtitle entries for matching",
            )

            segments = parse_segments_from_manus_response(response_text, all_subtitles)
            self._pipeline_state.selected_segments = segments

            self.app.call_from_thread(
                log.write_success,
                f"Loaded {len(segments)} segments from cached response",
            )
            self.app.call_from_thread(self._display_segments, segments)

            if self._pipeline_state.auto_segment_review:
                self.app.call_from_thread(self._proceed_to_cutting_screen)

        except Exception as error:
            logger.exception("Failed to load cached Manus response")
            self.app.call_from_thread(
                log.write_error, f"Failed to load cached response: {error}"
            )
            self.app.call_from_thread(self._re_enable_source_controls)

    def _re_enable_source_controls(self) -> None:
        """Re-enable all source selection controls after a failure."""
        self.query_one("#use-selected-button", Button).disabled = (self._selected_source_index < 0)
        self.query_one("#new-manus-button", Button).disabled = False
        self.query_one("#resume-task-button", Button).disabled = False
        self.query_one("#screen-subtitle", Label).update(
            "Load failed -- select another response or request new analysis"
        )

    @work(thread=True)
    def _resume_manus_task(self, task_url_or_id: str) -> None:
        """Resume or fetch results from an existing Manus task by URL or ID."""
        from api.manus_client import ManusClient
        from pipeline.segment_selector import parse_segments_from_manus_response
        from pipeline.subtitle_parser import load_subtitle_file

        log = self._log_widget
        task_id = self._extract_task_id_from_url(task_url_or_id)

        manus_api_key = os.getenv("MANUS_API_KEY", "")
        if not manus_api_key:
            self.app.call_from_thread(log.write_error, "MANUS_API_KEY not set in .env")
            return

        self.app.call_from_thread(
            log.write_step_header, f"Resuming Manus Task: {task_id}"
        )

        try:
            def on_progress(msg: str) -> None:
                self.app.call_from_thread(log.write_info, msg)

            client = ManusClient(api_key=manus_api_key)
            manus_response = client.fetch_task_response(task_id, on_progress)

            self._save_manus_response(manus_response)
            self._register_task(task_id, status="completed")
            self._pipeline_state.manus_task_id = task_id

            all_subtitles = load_subtitle_file(self._pipeline_state.subtitle_file_path)
            segments = parse_segments_from_manus_response(manus_response, all_subtitles)
            self._pipeline_state.selected_segments = segments

            self.app.call_from_thread(
                log.write_success, f"Loaded {len(segments)} segments from task {task_id}"
            )
            self.app.call_from_thread(self._display_segments, segments)

        except Exception as error:
            self.app.call_from_thread(log.write_error, str(error))
            self.app.call_from_thread(self._re_enable_source_controls)

    @staticmethod
    def _extract_task_id_from_url(url_or_id: str) -> str:
        """Extract task ID from a Manus URL or return raw ID."""
        url_or_id = url_or_id.strip()
        if "/" in url_or_id:
            return url_or_id.rstrip("/").split("/")[-1]
        return url_or_id

    # -- Task registry -- tracks all Manus tasks submitted from this folder --

    def _get_task_registry_path(self) -> Path | None:
        if not self._pipeline_state.video_file_path:
            return None
        return self._pipeline_state.video_file_path.parent / "manus_tasks.json"

    def _load_task_registry(self) -> list[dict]:
        registry_path = self._get_task_registry_path()
        if not registry_path or not registry_path.exists():
            return []
        try:
            data = json.loads(registry_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save_task_registry(self, tasks: list[dict]) -> None:
        registry_path = self._get_task_registry_path()
        if not registry_path:
            return
        registry_path.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

    def _register_task(self, task_id: str, status: str = "running") -> None:
        """Add a task to the local registry."""
        tasks = self._load_task_registry()
        for task in tasks:
            if task.get("task_id") == task_id:
                task["status"] = status
                self._save_task_registry(tasks)
                return
        tasks.append({
            "task_id": task_id,
            "submitted_at": datetime.now().isoformat(),
            "video_file": (
                self._pipeline_state.video_file_path.name
                if self._pipeline_state.video_file_path else ""
            ),
            "status": status,
        })
        self._save_task_registry(tasks)
        logger.info(f"Task registered: {task_id} ({status})")

    def _update_task_status(self, task_id: str, status: str) -> None:
        tasks = self._load_task_registry()
        for task in tasks:
            if task.get("task_id") == task_id:
                task["status"] = status
                break
        self._save_task_registry(tasks)

    def _proceed_to_cutting_screen(self) -> None:
        from tui.screens.cutting_screen import CuttingScreen
        self.app.push_screen(CuttingScreen(self._pipeline_state))

    def action_quit(self) -> None:
        self.app.exit()
