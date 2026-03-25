"""
tui/screens/file_selection_screen.py

First screen in the pipeline. The user selects the source video file.
Audio tracks are auto-discovered from the "Audio RAW" subfolder.
If no audio tracks are found, a manual audio file picker is shown as fallback.

Proceeds directly to the transcription screen (alignment is handled by
baking offset silence into the audio before sending to WhisperX).
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Rule

from models.pipeline_state import PipelineState
from pipeline.user_prefs import get_last_browse_directory, set_last_browse_directory
from tui.widgets.labeled_file_picker import LabeledFilePicker

VIDEO_FILE_PICKER_ID = "video-file-picker"
AUDIO_FILE_PICKER_ID = "audio-file-picker"

VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv", ".avi"]
AUDIO_EXTENSIONS = [".m4a", ".mp3", ".wav", ".aac", ".flac"]


class FileSelectionScreen(Screen):
    """Screen for selecting the source video file. Audio tracks are auto-discovered."""

    BINDINGS = [("q", "quit", "Quit")]

    DEFAULT_CSS = """
    FileSelectionScreen {
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
    #discovered-tracks-label {
        color: $success;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }
    #proceed-button {
        margin-top: 2;
        width: 100%;
    }
    #validation-message {
        color: $warning;
        margin-top: 1;
    }
    """

    def __init__(self, pipeline_state: PipelineState) -> None:
        super().__init__()
        self._pipeline_state = pipeline_state

    def compose(self) -> ComposeResult:
        last_directory = get_last_browse_directory()
        yield Header()
        with Vertical():
            yield Label("Sermon Shorts -- Phase 1", id="screen-title")
            yield Label("Select your sermon video to begin", id="screen-subtitle")
            yield Rule()
            yield LabeledFilePicker(
                label_text="Source Video File",
                allowed_extensions=VIDEO_EXTENSIONS,
                picker_id=VIDEO_FILE_PICKER_ID,
                placeholder_text="Select the full sermon video (e.g. sermon.mp4)",
                start_directory=last_directory,
            )
            yield Label("", id="discovered-tracks-label")
            yield LabeledFilePicker(
                label_text="Audio File (manual fallback)",
                allowed_extensions=AUDIO_EXTENSIONS,
                picker_id=AUDIO_FILE_PICKER_ID,
                placeholder_text="Select audio file manually",
                start_directory=last_directory,
            )
            yield Label("", id="validation-message")
            yield Button(
                "Proceed to Transcription ->",
                id="proceed-button",
                variant="primary",
                disabled=True,
            )
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(f"#{AUDIO_FILE_PICKER_ID}", LabeledFilePicker).display = False

    def on_labeled_file_picker_file_selected(
        self, event: LabeledFilePicker.FileSelected
    ) -> None:
        selected_directory = event.selected_file_path.parent
        set_last_browse_directory(selected_directory)

        if event.picker_id == VIDEO_FILE_PICKER_ID:
            self._pipeline_state.video_file_path = event.selected_file_path
            self._try_auto_discover_audio_tracks()
        elif event.picker_id == AUDIO_FILE_PICKER_ID:
            from pipeline.language_detect import detect_language_from_filename
            language_code = detect_language_from_filename(event.selected_file_path)
            self._pipeline_state.discovered_audio_tracks[language_code] = event.selected_file_path

        self._update_proceed_button_state()

    def _try_auto_discover_audio_tracks(self) -> None:
        from pipeline.audio_track_discovery import (
            discover_audio_tracks,
            format_discovered_tracks_summary,
        )
        tracks = discover_audio_tracks(self._pipeline_state.video_file_path)
        self._pipeline_state.discovered_audio_tracks = tracks

        tracks_label = self.query_one("#discovered-tracks-label", Label)
        audio_picker = self.query_one(f"#{AUDIO_FILE_PICKER_ID}", LabeledFilePicker)

        if tracks:
            tracks_label.update(format_discovered_tracks_summary(tracks))
            audio_picker.display = False
        else:
            tracks_label.update("No audio tracks found in Audio RAW -- select manually below")
            audio_picker.display = True
            audio_picker.start_directory = self._pipeline_state.video_file_path.parent

    def _update_proceed_button_state(self) -> None:
        can_proceed = self._pipeline_state.is_ready_for_transcription()
        proceed_button = self.query_one("#proceed-button", Button)
        proceed_button.disabled = not can_proceed

        validation_label = self.query_one("#validation-message", Label)
        if can_proceed:
            validation_label.update("")
        elif self._pipeline_state.video_file_path is None:
            validation_label.update("Select a video file to continue")
        else:
            validation_label.update("No audio tracks found -- select audio manually")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "proceed-button":
            self._proceed_to_transcription_screen()

    def _proceed_to_transcription_screen(self) -> None:
        from tui.screens.transcription_screen import TranscriptionScreen
        self.app.push_screen(TranscriptionScreen(self._pipeline_state))

    def action_quit(self) -> None:
        self.app.exit()
