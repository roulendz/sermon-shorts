"""
tui/screens/audio_alignment_screen.py

Sets the time offset between the separate audio file and the video.
Defaults to manual entry. Optional auto-detect via cross-correlation.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Input, Label, Rule
from textual import work

from models.pipeline_state import PipelineState
from tui.widgets.pipeline_log import PipelineLog


class AudioAlignmentScreen(Screen):
    """Screen for setting the audio-to-video time offset."""

    BINDINGS = [("q", "quit", "Quit")]

    DEFAULT_CSS = """
    AudioAlignmentScreen {
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
    #offset-explanation {
        color: $text-muted;
        margin-bottom: 1;
    }
    .offset-label {
        margin-top: 1;
        margin-bottom: 0;
    }
    #manual-offset-input {
        width: 20;
    }
    #apply-manual-offset-button {
        width: 12;
        margin-left: 1;
    }
    #auto-detect-button {
        margin-top: 1;
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
            yield Label("Step 1 -- Audio Alignment", id="screen-title")
            yield Label(
                "Set the offset between your audio recording and the video.",
                id="screen-subtitle",
            )
            yield Rule()
            yield Label(
                "How many seconds into the video does the audio recording start?\n"
                "Video has 10s intro before audio begins -> enter 10\n"
                "Audio and video start at the same time -> enter 0",
                id="offset-explanation",
            )
            yield Label("Offset in seconds:", classes="offset-label")
            with Horizontal():
                yield Input(
                    placeholder="e.g. 10",
                    value="0",
                    id="manual-offset-input",
                    type="number",
                )
                yield Button(
                    "Apply",
                    id="apply-manual-offset-button",
                    variant="default",
                )
            yield Rule()
            yield Button(
                "Try Auto-Detect (may not work with multilingual audio)",
                id="auto-detect-button",
                variant="warning",
            )
            yield PipelineLog(id="alignment-log")
            yield Button(
                "Proceed to Transcription ->",
                id="proceed-button",
                variant="primary",
            )
        yield Footer()

    def on_mount(self) -> None:
        self._log_widget = self.query_one("#alignment-log", PipelineLog)
        self._pipeline_state.audio_to_video_offset_seconds = 0.0
        if self._pipeline_state.auto_alignment:
            self._log_widget.write_info("Auto alignment enabled -- using offset 0.0s")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-manual-offset-button":
            self._apply_manual_offset_from_input()
        elif event.button.id == "auto-detect-button":
            self._run_alignment_detection()
        elif event.button.id == "proceed-button":
            self._apply_manual_offset_from_input()
            self._proceed_to_transcription_screen()

    def _apply_manual_offset_from_input(self) -> None:
        manual_input = self.query_one("#manual-offset-input", Input)
        try:
            offset_seconds = float(manual_input.value or "0")
            self._pipeline_state.audio_to_video_offset_seconds = offset_seconds
            self._log_widget.write_info(f"Offset set to {offset_seconds:.3f}s")
        except ValueError:
            self._log_widget.write_error("Invalid offset -- enter a number like 10 or 45.3")

    @work(thread=True)
    def _run_alignment_detection(self) -> None:
        from pipeline.audio_alignment import detect_audio_offset_seconds

        log = self._log_widget
        self.app.call_from_thread(log.write_step_header, "Cross-correlation audio alignment")
        self.app.call_from_thread(
            log.write_info,
            f"Video: {self._pipeline_state.video_file_path.name}",
        )
        self.app.call_from_thread(
            log.write_info,
            f"Audio: {self._pipeline_state.audio_file_path.name}",
        )

        def on_progress(msg: str) -> None:
            self.app.call_from_thread(log.write_info, msg)

        try:
            offset_seconds = detect_audio_offset_seconds(
                video_file_path=str(self._pipeline_state.video_file_path),
                separate_audio_file_path=str(self._pipeline_state.audio_file_path),
                on_progress=on_progress,
            )
            self.app.call_from_thread(
                log.write_success,
                f"Offset detected: {offset_seconds:.3f} seconds",
            )
            self.app.call_from_thread(self._set_detected_offset, offset_seconds)

        except Exception as error:
            self.app.call_from_thread(log.write_error, f"{type(error).__name__}: {error}")
            self.app.call_from_thread(
                log.write_info,
                "Enter the offset manually above.",
            )

    def _set_detected_offset(self, offset_seconds: float) -> None:
        self._pipeline_state.audio_to_video_offset_seconds = offset_seconds
        manual_input = self.query_one("#manual-offset-input", Input)
        manual_input.value = str(offset_seconds)

    def _proceed_to_transcription_screen(self) -> None:
        from tui.screens.transcription_screen import TranscriptionScreen
        self.app.push_screen(TranscriptionScreen(self._pipeline_state))

    def action_quit(self) -> None:
        self.app.exit()
