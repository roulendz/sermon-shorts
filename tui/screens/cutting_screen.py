"""
tui/screens/cutting_screen.py

Cuts each selected segment from the source video using FFmpeg.
Saves each segment as a .mp4 file with an accompanying .srt subtitle file.
Optionally crops segments to 9:16 portrait with face tracking.
Detects existing clip folders and offers reuse or recut options.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Rule,
)
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
    #options-container {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $primary-darken-2;
    }
    #options-label {
        text-style: bold;
        margin-bottom: 0;
    }
    #existing-clips-container {
        height: auto;
        margin-bottom: 1;
        padding: 1;
        border: solid $warning-darken-1;
    }
    #existing-clips-label {
        text-style: bold;
        color: $warning;
        margin-bottom: 0;
    }
    #existing-clips-radio {
        margin-top: 0;
        height: auto;
    }
    #use-existing-button {
        margin-top: 1;
        width: 100%;
    }
    #recut-button {
        margin-top: 0;
        width: 100%;
    }
    #start-cutting-button {
        margin-top: 1;
        width: 100%;
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
        self._existing_clip_directories: list[Path] = []
        self._selected_existing_directory: Optional[Path] = None
        self._enable_debug_overlay: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical():
            yield Label("Step 4 — Cutting Segments", id="screen-title")
            yield Label(
                f"{len(self._pipeline_state.selected_segments)} segments ready to cut",
                id="screen-subtitle",
            )
            yield Rule()
            with Vertical(id="options-container"):
                yield Label("Output Options", id="options-label")
                yield Checkbox(
                    "Crop to 9:16 portrait with face tracking (TikTok / Shorts / Reels)",
                    value=self._pipeline_state.enable_portrait_crop,
                    id="portrait-crop-checkbox",
                )
                yield Checkbox(
                    "Generate debug overlay video (shows tracking data on original)",
                    value=False,
                    id="debug-overlay-checkbox",
                )
                yield Label("Playback speed multiplier (1.0 = normal):")
                yield Input(
                    value=str(self._pipeline_state.portrait_speed_multiplier),
                    placeholder="1.3",
                    id="speed-multiplier-input",
                )
            with Vertical(id="existing-clips-container"):
                yield Label("Existing Clips Found", id="existing-clips-label")
                yield RadioSet(id="existing-clips-radio")
                yield Button(
                    "Use Existing Clips",
                    id="use-existing-button",
                    variant="warning",
                )
                yield Button(
                    "Recut (new folder)",
                    id="recut-button",
                    variant="default",
                )
            yield Button(
                "Start Cutting",
                id="start-cutting-button",
                variant="primary",
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
        self._existing_clip_directories = self._find_existing_clip_directories()

        existing_container = self.query_one("#existing-clips-container")
        start_button = self.query_one("#start-cutting-button", Button)

        if self._existing_clip_directories:
            start_button.display = False
            radio_set = self.query_one("#existing-clips-radio", RadioSet)
            for clip_directory in self._existing_clip_directories:
                all_mp4s = list(clip_directory.glob("*.mp4"))
                mp4_count = len([p for p in all_mp4s if "[portrait]" not in p.name and "[debug]" not in p.name])
                label = f"{clip_directory.name}/ ({mp4_count} clips)"
                radio_set.mount(RadioButton(label))
            if self._existing_clip_directories:
                self._selected_existing_directory = self._existing_clip_directories[0]
        else:
            existing_container.display = False

        if self._pipeline_state.auto_video_cutting and not self._existing_clip_directories:
            self._begin_cutting()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "existing-clips-radio":
            selected_index = event.index
            if 0 <= selected_index < len(self._existing_clip_directories):
                self._selected_existing_directory = self._existing_clip_directories[selected_index]

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "portrait-crop-checkbox":
            self._pipeline_state.enable_portrait_crop = event.value
        elif event.checkbox.id == "debug-overlay-checkbox":
            self._enable_debug_overlay = event.value

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "speed-multiplier-input":
            try:
                value = float(event.value)
                if 0.5 <= value <= 3.0:
                    self._pipeline_state.portrait_speed_multiplier = value
            except ValueError:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-cutting-button":
            self._begin_cutting()
        elif event.button.id == "recut-button":
            self._begin_cutting()
        elif event.button.id == "use-existing-button":
            self._use_existing_clips()
        elif event.button.id == "open-output-button":
            self._open_output_directory_in_file_explorer()

    def _find_existing_clip_directories(self) -> list[Path]:
        """Find all existing clip directories that contain .mp4 files."""
        clips_base = self._pipeline_state.video_file_path.parent / "clips"
        if not clips_base.exists():
            return []

        found_directories: list[Path] = []

        has_root_mp4s = any(clips_base.glob("*.mp4"))
        if has_root_mp4s:
            found_directories.append(clips_base)

        for item in sorted(clips_base.iterdir(), reverse=True):
            if item.is_dir() and re.match(r"v\d+-", item.name):
                if any(item.glob("*.mp4")):
                    found_directories.append(item)

        return found_directories

    def _use_existing_clips(self) -> None:
        """Use already-cut clips from an existing directory."""
        if not self._selected_existing_directory:
            return

        self._disable_all_controls()
        self._clips_directory = self._selected_existing_directory
        existing_mp4s = sorted(self._clips_directory.glob("*.mp4"))
        portrait_already = [p for p in existing_mp4s if "[portrait]" in p.name]
        landscape_clips = [p for p in existing_mp4s if "[portrait]" not in p.name]

        log = self._log_widget
        log.write_step_header("Using Existing Clips")
        log.write_info(f"Directory: {self._clips_directory}")
        log.write_info(f"Found {len(landscape_clips)} landscape clips")

        if self._pipeline_state.enable_portrait_crop and landscape_clips:
            segments = self._pipeline_state.selected_segments
            clip_segment_pairs: list[tuple[Path, VideoSegment]] = []

            for clip_path in landscape_clips:
                matched_segment = self._match_clip_to_segment(clip_path, segments)
                if matched_segment:
                    clip_segment_pairs.append((clip_path, matched_segment))
                else:
                    clip_segment_pairs.append((clip_path, self._make_placeholder_segment(clip_path)))

            self._run_portrait_cropping_on_existing(clip_segment_pairs)
        else:
            self._show_completion_message(len(landscape_clips), len(landscape_clips))

    def _match_clip_to_segment(
        self,
        clip_path: Path,
        segments: list[VideoSegment],
    ) -> Optional[VideoSegment]:
        """Try to match a clip filename to a segment by title."""
        clip_stem = clip_path.stem.lower()
        for segment in segments:
            if segment.suggested_title and segment.suggested_title.lower() in clip_stem:
                return segment
        return None

    def _make_placeholder_segment(self, clip_path: Path) -> VideoSegment:
        """Create a placeholder segment for duration calculation from clip file."""
        import subprocess
        from datetime import timedelta

        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(clip_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            duration = float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError):
            duration = 60.0

        return VideoSegment(
            index=0,
            start_time=timedelta(seconds=0),
            end_time=timedelta(seconds=duration),
            transcript_text="",
            selection_reason="",
        )

    def _disable_all_controls(self) -> None:
        for button_id in ("start-cutting-button", "recut-button", "use-existing-button"):
            try:
                self.query_one(f"#{button_id}", Button).disabled = True
            except Exception:
                pass
        try:
            self.query_one("#portrait-crop-checkbox", Checkbox).disabled = True
        except Exception:
            pass

    def _begin_cutting(self) -> None:
        self._disable_all_controls()
        self._clips_directory = self._resolve_new_clips_directory()
        self._clips_directory.mkdir(parents=True, exist_ok=True)
        self._run_segment_cutting()

    def _resolve_new_clips_directory(self) -> Path:
        """
        Determine clips output directory with versioning.
        First run:  {video_folder}/clips/
        If /clips/ already has files: {video_folder}/clips/v2-YYYY-MM-DD/
        """
        clips_base = self._pipeline_state.video_file_path.parent / "clips"
        today = date.today().isoformat()

        if not clips_base.exists():
            return clips_base

        has_files = any(clips_base.glob("*.mp4"))
        if not has_files:
            subdirs = [d for d in clips_base.iterdir() if d.is_dir() and re.match(r"v\d+-", d.name)]
            if not subdirs:
                return clips_base

        max_version = 1
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
            find_end_of_preceding_subtitle,
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

        cut_video_paths: list[tuple[Path, VideoSegment]] = []
        successfully_cut_count = 0

        for segment in segments:
            try:
                preceding_subtitle_end = find_end_of_preceding_subtitle(
                    all_subtitles, segment.start_time,
                )
                if preceding_subtitle_end is not None and preceding_subtitle_end != segment.start_time:
                    original_start_timestamp = segment.ffmpeg_start_timestamp
                    segment.start_time = preceding_subtitle_end
                    self.app.call_from_thread(
                        log.write_info,
                        f"Segment {segment.index}: snapped start "
                        f"{original_start_timestamp} → {segment.ffmpeg_start_timestamp} "
                        f"(aligned to preceding subtitle boundary)",
                    )

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
                cut_video_paths.append((video_output_path, segment))
                successfully_cut_count += 1

            except Exception as error:
                self.app.call_from_thread(
                    log.write_error,
                    f"Segment {segment.index} failed: {error}",
                )

        if self._pipeline_state.enable_portrait_crop and cut_video_paths:
            self._run_portrait_cropping(cut_video_paths)
        else:
            self.app.call_from_thread(
                self._show_completion_message,
                successfully_cut_count,
                len(segments),
            )

    def _run_portrait_cropping(
        self,
        cut_video_paths: list[tuple[Path, VideoSegment]],
    ) -> None:
        from pipeline.face_cropper import (
            crop_segment_to_portrait,
            generate_debug_video,
            build_portrait_output_path,
            build_debug_output_path,
            PortraitCropError,
        )

        log = self._log_widget
        model_directory = Path(__file__).resolve().parent.parent.parent / ".models"

        self.app.call_from_thread(log.write_step_header, "Portrait Pose-Tracked Cropping")
        self.app.call_from_thread(
            log.write_info,
            f"Processing {len(cut_video_paths)} segments with face tracking...",
        )

        portrait_success_count = 0

        for landscape_path, segment in cut_video_paths:
            try:
                if self._enable_debug_overlay:
                    debug_path = build_debug_output_path(landscape_path)
                    self.app.call_from_thread(
                        log.write_info,
                        f"Segment {segment.index}: generating debug overlay...",
                    )
                    generate_debug_video(
                        source_video_path=landscape_path,
                        output_file_path=debug_path,
                        start_seconds=0.0,
                        duration_seconds=segment.duration_seconds,
                        model_directory=model_directory,
                    )
                    self.app.call_from_thread(
                        log.write_success,
                        f"Segment {segment.index} debug: {debug_path.name}",
                    )

                portrait_output_path = build_portrait_output_path(landscape_path)
                self.app.call_from_thread(
                    log.write_info,
                    f"Segment {segment.index}: cropping to portrait...",
                )

                def on_crop_progress(percent: float, seg_index=segment.index):
                    self.app.call_from_thread(
                        log.write_info,
                        f"  Segment {seg_index}: {percent:.0f}% encoded",
                    )

                crop_segment_to_portrait(
                    source_video_path=landscape_path,
                    output_file_path=portrait_output_path,
                    start_seconds=0.0,
                    duration_seconds=segment.duration_seconds,
                    model_directory=model_directory,
                    speed_multiplier=self._pipeline_state.portrait_speed_multiplier,
                    on_progress=on_crop_progress,
                )

                speed_label = ""
                if self._pipeline_state.portrait_speed_multiplier != 1.0:
                    speed_label = f" ({self._pipeline_state.portrait_speed_multiplier}x)"
                self.app.call_from_thread(
                    log.write_success,
                    f"Segment {segment.index} portrait{speed_label}: {portrait_output_path.name}",
                )
                portrait_success_count += 1

            except (PortraitCropError, Exception) as error:
                self.app.call_from_thread(
                    log.write_error,
                    f"Segment {segment.index} portrait crop failed: {error}",
                )

        total_segments = len(self._pipeline_state.selected_segments)
        self.app.call_from_thread(
            self._show_completion_message,
            len(cut_video_paths),
            total_segments,
            portrait_success_count,
        )

    @work(thread=True)
    def _run_portrait_cropping_on_existing(
        self,
        clip_segment_pairs: list[tuple[Path, VideoSegment]],
    ) -> None:
        from pipeline.face_cropper import (
            crop_segment_to_portrait,
            generate_debug_video,
            build_portrait_output_path,
            build_debug_output_path,
            PortraitCropError,
        )

        log = self._log_widget
        model_directory = Path(__file__).resolve().parent.parent.parent / ".models"

        self.app.call_from_thread(log.write_step_header, "Portrait Pose-Tracked Cropping")
        self.app.call_from_thread(
            log.write_info,
            f"Processing {len(clip_segment_pairs)} existing clips with face tracking...",
        )

        portrait_success_count = 0

        for clip_path, segment in clip_segment_pairs:
            if "[portrait]" in clip_path.name or "[debug]" in clip_path.name:
                continue

            try:
                if self._enable_debug_overlay:
                    debug_path = build_debug_output_path(clip_path)
                    self.app.call_from_thread(
                        log.write_info,
                        f"  Debug overlay: {clip_path.name}",
                    )
                    generate_debug_video(
                        source_video_path=clip_path,
                        output_file_path=debug_path,
                        start_seconds=0.0,
                        duration_seconds=segment.duration_seconds,
                        model_directory=model_directory,
                    )
                    self.app.call_from_thread(
                        log.write_success,
                        f"  Debug: {debug_path.name}",
                    )

                portrait_output_path = build_portrait_output_path(clip_path)
                if portrait_output_path.exists() and not self._enable_debug_overlay:
                    self.app.call_from_thread(
                        log.write_info,
                        f"  Skipping (already exists): {portrait_output_path.name}",
                    )
                    portrait_success_count += 1
                    continue

                self.app.call_from_thread(
                    log.write_info,
                    f"  Cropping: {clip_path.name}",
                )

                def on_crop_progress(percent: float, name=clip_path.stem):
                    self.app.call_from_thread(
                        log.write_info,
                        f"    {name}: {percent:.0f}% encoded",
                    )

                crop_segment_to_portrait(
                    source_video_path=clip_path,
                    output_file_path=portrait_output_path,
                    start_seconds=0.0,
                    duration_seconds=segment.duration_seconds,
                    model_directory=model_directory,
                    speed_multiplier=self._pipeline_state.portrait_speed_multiplier,
                    on_progress=on_crop_progress,
                )

                speed_label = ""
                if self._pipeline_state.portrait_speed_multiplier != 1.0:
                    speed_label = f" ({self._pipeline_state.portrait_speed_multiplier}x)"
                self.app.call_from_thread(
                    log.write_success,
                    f"  Portrait created{speed_label}: {portrait_output_path.name}",
                )
                portrait_success_count += 1

            except (PortraitCropError, Exception) as error:
                self.app.call_from_thread(
                    log.write_error,
                    f"  Portrait crop failed for {clip_path.name}: {error}",
                )

        self.app.call_from_thread(
            self._show_completion_message,
            len(clip_segment_pairs),
            len(clip_segment_pairs),
            portrait_success_count,
        )

    def _save_segment_subtitles(
        self,
        all_subtitles,
        segment: VideoSegment,
        subtitle_output_path: Path,
        save_function,
        slice_function,
    ) -> None:
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
        portrait_count: int = 0,
    ) -> None:
        done_label = self.query_one("#done-label", Label)
        message = f"Done! {successfully_cut_count}/{total_segment_count} segments cut successfully."
        if portrait_count > 0:
            message += f" {portrait_count} portrait versions created."
        done_label.update(message)

        open_output_button = self.query_one("#open-output-button", Button)
        open_output_button.disabled = False

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
