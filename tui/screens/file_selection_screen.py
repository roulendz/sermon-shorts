"""
tui/screens/file_selection_screen.py

First screen in the pipeline. The user selects:
  - The source video file (large, used for cutting)
  - The audio file (smaller, sent to WhisperX for transcription)

Proceeds to the audio alignment screen once both files are selected.
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
    """Screen for selecting the source video and audio files."""

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
        last_dir = get_last_browse_directory()
        yield Header()
        with Vertical():
            yield Label("Sermon Shorts — Phase 1", id="screen-title")
            yield Label("Select your source files to begin", id="screen-subtitle")
            yield Rule()
            yield LabeledFilePicker(
                label_text="Source Video File",
                allowed_extensions=VIDEO_EXTENSIONS,
                picker_id=VIDEO_FILE_PICKER_ID,
                placeholder_text="Select the full sermon video (e.g. sermon.mp4)",
                start_directory=last_dir,
            )
            yield LabeledFilePicker(
                label_text="Audio File for Transcription",
                allowed_extensions=AUDIO_EXTENSIONS,
                picker_id=AUDIO_FILE_PICKER_ID,
                placeholder_text="Select the audio recording (e.g. sermon_audio.m4a)",
                start_directory=last_dir,
            )
            yield Label("", id="validation-message")
            yield Button(
                "Detect Audio Offset →",
                id="proceed-button",
                variant="primary",
                disabled=True,
            )
        yield Footer()

    def on_labeled_file_picker_file_selected(
        self, event: LabeledFilePicker.FileSelected
    ) -> None:
        selected_dir = event.selected_file_path.parent
        set_last_browse_directory(selected_dir)

        if event.picker_id == VIDEO_FILE_PICKER_ID:
            self._pipeline_state.video_file_path = event.selected_file_path
            # Set the audio picker to start in the same folder
            audio_picker = self.query_one(f"#{AUDIO_FILE_PICKER_ID}", LabeledFilePicker)
            audio_picker.start_directory = selected_dir
        elif event.picker_id == AUDIO_FILE_PICKER_ID:
            self._pipeline_state.audio_file_path = event.selected_file_path

        self._update_proceed_button_state()

    def _update_proceed_button_state(self) -> None:
        both_files_selected = self._pipeline_state.is_ready_for_alignment()
        proceed_button = self.query_one("#proceed-button", Button)
        proceed_button.disabled = not both_files_selected

        validation_label = self.query_one("#validation-message", Label)
        if both_files_selected:
            validation_label.update("")
        else:
            validation_label.update("Select both files to continue")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "proceed-button":
            self._proceed_to_alignment_screen()

    def _proceed_to_alignment_screen(self) -> None:
        from tui.screens.audio_alignment_screen import AudioAlignmentScreen
        self.app.push_screen(AudioAlignmentScreen(self._pipeline_state))

    def action_quit(self) -> None:
        self.app.exit()
