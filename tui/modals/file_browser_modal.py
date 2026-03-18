"""
tui/modals/file_browser_modal.py

A keyboard-navigable terminal file browser modal.
On Windows, shows a drive picker first, then loads the directory tree.
Dismisses with the selected file path, or None if cancelled.
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Label


def _get_windows_drives() -> list[Path]:
    """Return available drive roots on Windows."""
    drives = []
    for letter in string.ascii_uppercase:
        drive = Path(f"{letter}:\\")
        if drive.exists():
            drives.append(drive)
    return drives


class FileBrowserModal(ModalScreen[Path | None]):
    """
    Modal that presents a navigable directory tree for file selection.
    On Windows without a start_directory, shows drive buttons first.
    """

    BINDINGS = [
        Binding("escape", "cancel_selection", "Cancel"),
        Binding("backspace", "go_up", "Go Up"),
    ]

    DEFAULT_CSS = """
    FileBrowserModal {
        align: center middle;
    }
    #modal-container {
        width: 70;
        height: 30;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    #modal-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    #allowed-extensions-hint {
        color: $text-muted;
        margin-bottom: 0;
    }
    #nav-bar {
        height: 3;
        margin-bottom: 0;
    }
    #current-path-label {
        width: 1fr;
        content-align: left middle;
        color: $text-muted;
        height: 3;
        padding: 0 1;
    }
    #go-up-button {
        min-width: 10;
        margin-left: 1;
    }
    #drive-bar {
        height: auto;
        margin-bottom: 1;
    }
    #drive-prompt {
        margin-bottom: 1;
    }
    .drive-button {
        min-width: 6;
        margin: 0 1 0 0;
    }
    DirectoryTree {
        height: 1fr;
        border: solid $primary-darken-2;
    }
    #cancel-hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    def __init__(
        self,
        modal_title: str,
        allowed_extensions: list[str],
        start_directory: Path | None = None,
    ) -> None:
        super().__init__()
        self._modal_title = modal_title
        self._allowed_extensions = [ext.lower() for ext in allowed_extensions]
        self._start_directory = start_directory
        self._is_windows = sys.platform == "win32"
        self._show_drive_picker = self._is_windows and start_directory is None
        self._current_root: Path | None = start_directory

    def compose(self) -> ComposeResult:
        extensions_display = ", ".join(self._allowed_extensions)
        with Vertical(id="modal-container"):
            yield Label(self._modal_title, id="modal-title")
            yield Label(f"Allowed: {extensions_display}", id="allowed-extensions-hint")
            if self._show_drive_picker:
                yield Label("Select a drive:", id="drive-prompt")
                with Horizontal(id="drive-bar"):
                    for drive in _get_windows_drives():
                        yield Button(
                            f"{drive.drive}",
                            id=f"drive-{drive.drive[0].lower()}",
                            classes="drive-button",
                        )
                # Nav bar and tree hidden until drive is picked
            else:
                start = self._start_directory or Path.home()
                self._current_root = start
                with Horizontal(id="nav-bar"):
                    yield Label(str(start), id="current-path-label")
                    yield Button(".. Up", id="go-up-button", variant="default")
                yield DirectoryTree(str(start), id="dir-tree")
            yield Label("Backspace = Go Up  |  Enter = Select  |  Escape = Cancel", id="cancel-hint")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("drive-"):
            event.stop()
            drive_letter = event.button.id.split("-")[1].upper()
            drive_path = Path(f"{drive_letter}:\\")
            # Remove drive picker widgets
            for widget in self.query("#drive-prompt, #drive-bar"):
                await widget.remove()
            # Mount nav bar and tree
            cancel_hint = self.query_one("#cancel-hint")
            container = self.query_one("#modal-container", Vertical)
            nav_bar = Horizontal(id="nav-bar")
            await container.mount(nav_bar, before=cancel_hint)
            await nav_bar.mount(Label(str(drive_path), id="current-path-label"))
            await nav_bar.mount(Button(".. Up", id="go-up-button", variant="default"))
            tree = DirectoryTree(str(drive_path), id="dir-tree")
            await container.mount(tree, before=cancel_hint)
            self._current_root = drive_path

        elif event.button.id == "go-up-button":
            event.stop()
            self._navigate_to_parent()

    def _navigate_to_parent(self) -> None:
        if self._current_root is None:
            return

        parent = self._current_root.parent

        # On Windows, if we're at the drive root (e.g. D:\), parent == D:\ — no point going up
        if parent == self._current_root:
            return

        self._current_root = parent
        try:
            tree = self.query_one("#dir-tree", DirectoryTree)
            tree.path = parent
            tree.reload()
            path_label = self.query_one("#current-path-label", Label)
            path_label.update(str(parent))
        except Exception:
            pass

    def action_go_up(self) -> None:
        self._navigate_to_parent()

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        selected_path = Path(event.path)
        if self._is_allowed_file_extension(selected_path):
            self.dismiss(selected_path)

    def _is_allowed_file_extension(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self._allowed_extensions

    def action_cancel_selection(self) -> None:
        self.dismiss(None)
