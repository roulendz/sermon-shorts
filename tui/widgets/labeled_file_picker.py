"""
tui/widgets/labeled_file_picker.py

A reusable widget that shows a label, the currently selected file path,
and a Browse button that opens the FileBrowserModal.

Posts a FileSelected message when the user picks a file.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Button, Label, Static


class LabeledFilePicker(Static):
    """
    Composite widget for file selection.
    Displays label + selected path + browse button.
    """

    DEFAULT_CSS = """
    LabeledFilePicker {
        height: auto;
        margin-bottom: 1;
    }
    .file-picker-label {
        text-style: bold;
        margin-bottom: 0;
    }
    .file-picker-row {
        height: 3;
    }
    .selected-path-display {
        width: 1fr;
        border: solid $primary-darken-2;
        padding: 0 1;
        color: $text-muted;
        height: 3;
        content-align: left middle;
    }
    .selected-path-display.path-selected {
        color: $success;
    }
    .browse-button {
        width: 12;
        margin-left: 1;
    }
    """

    class FileSelected(Message):
        """Posted when the user selects a file through this picker."""
        def __init__(self, picker_id: str, selected_file_path: Path) -> None:
            super().__init__()
            self.picker_id = picker_id
            self.selected_file_path = selected_file_path

    def __init__(
        self,
        label_text: str,
        allowed_extensions: list[str],
        picker_id: str,
        placeholder_text: str = "No file selected",
        start_directory: Path | None = None,
    ) -> None:
        super().__init__(id=picker_id)
        self._label_text = label_text
        self._allowed_extensions = allowed_extensions
        self._picker_id = picker_id
        self._placeholder_text = placeholder_text
        self._selected_file_path: Path | None = None
        self.start_directory: Path | None = start_directory

    def compose(self) -> ComposeResult:
        yield Label(self._label_text, classes="file-picker-label")
        with Horizontal(classes="file-picker-row"):
            yield Label(self._placeholder_text, id=f"{self._picker_id}-path-display", classes="selected-path-display")
            yield Button("Browse", id=f"{self._picker_id}-browse-button", classes="browse-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == f"{self._picker_id}-browse-button":
            event.stop()
            self._open_file_browser()

    def _open_file_browser(self) -> None:
        from tui.modals.file_browser_modal import FileBrowserModal

        def handle_file_selected(selected_path: Path | None) -> None:
            if selected_path is not None:
                self._update_selected_path(selected_path)

        self.app.push_screen(
            FileBrowserModal(
                modal_title=f"Select {self._label_text}",
                allowed_extensions=self._allowed_extensions,
                start_directory=self.start_directory,
            ),
            callback=handle_file_selected,
        )

    def _update_selected_path(self, selected_file_path: Path) -> None:
        self._selected_file_path = selected_file_path
        path_display = self.query_one(f"#{self._picker_id}-path-display", Label)
        path_display.update(str(selected_file_path))
        path_display.add_class("path-selected")
        self.post_message(self.FileSelected(self._picker_id, selected_file_path))

    @property
    def selected_file_path(self) -> Path | None:
        return self._selected_file_path
