"""
tui/screens/cutting_screen.py

Cuts each selected segment from the source video using FFmpeg.
Saves each segment as a .mp4 file with an accompanying .srt subtitle file.
Displays progress for each segment as it is cut.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Rule
from textual import work

from models.pipeline_state import PipelineState
from models.video_segment import VideoSegment
from tui.widgets.pipeline_log import PipelineLog


class CuttingScreen(Screen):
    """Screen that cuts video segments using FFmpeg and saves SRT sidecars."""

    BINDINGS = [("q", "quit", "Quit")]

    DEFAULT_CSS = """
    CuttingScreen {
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
    #done-label {
        text-style: bold;
        color: $success;
        margin-top: 1;
    }
    #open-output-button {
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
            yield Label("Step 4 — Cutting Segments", id="screen-title")
            yield Label(
                f"Cutting {len(self._pipeline_state.selected_segments)} segments from video...",
                id="screen-subtitle",
            )
            yield Rule()
            yield PipelineLog(id="cutting-log")
            yield Label("", id="done-label")
            yield Button(
                "Open Output Folder",
                id="open-output-button",
                variant="success",
                disabled=True,
            )
        yield Footer()

    def on_mount(self) -> None:
        self._log_widget = self.query_one("#cutting-log", PipelineLog)
        self._clips_directory = self._resolve_clips_directory()
        self._clips_directory.mkdir(parents=True, exist_ok=True)
        self._run_segment_cutting()

    def _resolve_clips_directory(self) -> Path:
        """
        Determine clips output directory with versioning.
        First run:  {video_folder}/clips/
        If /clips/ already has files: {video_folder}/clips/v2-YYYY-MM-DD/
        Next run:                      {video_folder}/clips/v3-YYYY-MM-DD/  etc.
        """
        import re
        from datetime import date

        clips_base = self._pipeline_state.video_file_path.parent / "clips"
        today = date.today().isoformat()  # YYYY-MM-DD

        if not clips_base.exists():
            return clips_base

        # Check if clips_base has any video files directly in it
        has_files = any(clips_base.glob("*.mp4"))
        if not has_files:
            # Check for versioned subdirs — if none, use clips_base directly
            subdirs = [d for d in clips_base.iterdir() if d.is_dir() and re.match(r"v\d+-", d.name)]
            if not subdirs:
                return clips_base

        # Find highest existing version number
        max_version = 1  # existing files in clips/ count as v1
        for item in clips_base.iterdir():
            if item.is_dir():
                match = re.match(r"v(\d+)-", item.name)
                if match:
                    max_version = max(max_version, int(match.group(1)))

        next_version = max_version + 1
        return clips_base / f"v{next_version}-{today}"

    @work(thread=True)
    def _run_segment_cutting(self) -> None:
        from pipeline.subtitle_parser import (
            load_subtitle_file,
            slice_subtitles_within_window,
            save_subtitles_to_file,
        )
        from pipeline.video_cutter import (
            cut_segment_from_video,
            build_output_video_file_path,
            build_output_subtitle_file_path,
        )

        log = self._log_widget
        segments = self._pipeline_state.selected_segments

        self.app.call_from_thread(log.write_step_header, "FFmpeg Segment Cutting")
        self.app.call_from_thread(
            log.write_info,
            f"Output directory: {self._clips_directory}",
        )

        try:
            all_subtitles = load_subtitle_file(self._pipeline_state.subtitle_file_path)
        except Exception as error:
            self.app.call_from_thread(log.write_error, f"Could not load subtitle file: {error}")
            return

        successfully_cut_count = 0

        for segment in segments:
            try:
                self.app.call_from_thread(
                    log.write_info,
                    f"Cutting segment {segment.index}: "
                    f"{segment.ffmpeg_start_timestamp} → {segment.ffmpeg_end_timestamp}",
                )

                video_path = self._pipeline_state.video_file_path
                video_output_path = build_output_video_file_path(
                    self._clips_directory, segment, video_path,
                )
                cut_segment_from_video(
                    source_video_path=video_path,
                    segment=segment,
                    output_file_path=video_output_path,
                )

                self._save_segment_subtitles(
                    all_subtitles=all_subtitles,
                    segment=segment,
                    subtitle_output_path=build_output_subtitle_file_path(
                        self._clips_directory, segment, video_path,
                    ),
                    save_function=save_subtitles_to_file,
                    slice_function=slice_subtitles_within_window,
                )

                self.app.call_from_thread(
                    log.write_success,
                    f"Segment {segment.index} saved: {video_output_path.name}",
                )
                successfully_cut_count += 1

            except Exception as error:
                self.app.call_from_thread(
                    log.write_error,
                    f"Segment {segment.index} failed: {error}",
                )

        self.app.call_from_thread(
            self._show_completion_message,
            successfully_cut_count,
            len(segments),
        )

    def _save_segment_subtitles(
        self,
        all_subtitles,
        segment: VideoSegment,
        subtitle_output_path: Path,
        save_function,
        slice_function,
    ) -> None:
        from datetime import timedelta
        sliced_subtitles = slice_function(
            all_subtitles,
            window_start=segment.start_time,
            window_end=segment.end_time,
        )
        save_function(sliced_subtitles, subtitle_output_path)

    def _show_completion_message(
        self,
        successfully_cut_count: int,
        total_segment_count: int,
    ) -> None:
        done_label = self.query_one("#done-label", Label)
        done_label.update(
            f"Done! {successfully_cut_count}/{total_segment_count} segments cut successfully."
        )

        open_output_button = self.query_one("#open-output-button", Button)
        open_output_button.disabled = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-output-button":
            self._open_output_directory_in_file_explorer()

    def _open_output_directory_in_file_explorer(self) -> None:
        import platform
        import subprocess

        output_path = str(self._clips_directory.resolve())

        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", output_path])
        elif system == "Windows":
            subprocess.Popen(["explorer", output_path])
        else:
            subprocess.Popen(["xdg-open", output_path])

    def action_quit(self) -> None:
        self.app.exit()
