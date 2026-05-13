"""
main.py — Entry point for the Sermon Shorts pipeline TUI.
"""

from dotenv import load_dotenv
load_dotenv()

from pipeline.logging_config import configure_file_logging
configure_file_logging()

from textual.app import App

from models.pipeline_state import PipelineState
from tui.screens.file_selection_screen import FileSelectionScreen


class SermonShortsApp(App):
    TITLE = "Sermon Shorts"

    def __init__(self) -> None:
        super().__init__()
        self._pipeline_state = PipelineState()

    def on_mount(self) -> None:
        self.push_screen(FileSelectionScreen(self._pipeline_state))


if __name__ == "__main__":
    app = SermonShortsApp()
    app.run()
