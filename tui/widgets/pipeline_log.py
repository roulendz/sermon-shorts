"""
tui/widgets/pipeline_log.py

A scrollable log widget for displaying real-time pipeline progress.
Wraps Textual's Log widget with helper methods for common message types.
"""

from __future__ import annotations

from textual.widgets import Log


class PipelineLog(Log):
    """
    Scrollable log widget with helper methods for pipeline status messages.
    Automatically scrolls to the latest message.
    """

    DEFAULT_CSS = """
    PipelineLog {
        border: solid $primary-darken-2;
        min-height: 10;
        height: 1fr;
        max-height: 100%;
        background: $surface-darken-1;
        scrollbar-color: $primary;
        scrollbar-color-hover: $primary-lighten-2;
    }
    """

    def write_info(self, message: str) -> None:
        self.write_line(f"  {message}")

    def write_success(self, message: str) -> None:
        self.write_line(f"[OK] {message}")

    def write_error(self, message: str) -> None:
        self.write_line(f"[ERR] {message}")

    def write_step_header(self, step_name: str) -> None:
        self.write_line("")
        self.write_line(f"--- {step_name} ---")
