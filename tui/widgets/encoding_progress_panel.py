"""
tui/widgets/encoding_progress_panel.py

Progress panel showing current encoding clip, progress bar, ETA, and overall count.
Designed for two-column layout alongside PipelineLog.
"""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Label, ProgressBar


class EncodingProgressPanel(Vertical):

    DEFAULT_CSS = """
    EncodingProgressPanel {
        border: solid $accent-darken-2;
        padding: 1;
        height: 100%;
        width: 1fr;
    }
    #progress-step-name {
        text-style: bold;
        color: $accent;
    }
    #progress-clip-name {
        margin-top: 1;
    }
    #encoding-progress-bar {
        margin-top: 1;
    }
    #progress-detail {
        margin-top: 0;
    }
    #progress-overall {
        margin-top: 1;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._encoding_start_time: float = 0.0
        self._step_name: str = ""
        self._total_clips: int = 0

    def compose(self) -> ComposeResult:
        yield Label("Encoding Progress", id="progress-step-name")
        yield Label("Waiting for encoding...", id="progress-clip-name")
        yield ProgressBar(total=100, show_eta=False, id="encoding-progress-bar")
        yield Label("", id="progress-detail")
        yield Label("", id="progress-overall")

    def _format_duration(self, total_seconds: float) -> str:
        if total_seconds >= 60:
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)
            return f"{minutes}m {seconds}s"
        return f"{total_seconds:.0f}s"

    def start_step(self, step_name: str, total_clips: int) -> None:
        self._step_name = step_name
        self._total_clips = total_clips
        self.query_one("#progress-step-name", Label).update(step_name)
        self.query_one("#progress-clip-name", Label).update("Starting...")
        self.query_one("#encoding-progress-bar", ProgressBar).update(
            total=100, progress=0,
        )
        self.query_one("#progress-detail", Label).update("")
        self.query_one("#progress-overall", Label).update(
            f"Clip 0 of {total_clips}"
        )

    def start_clip(self, clip_name: str, clip_number: int) -> None:
        self._encoding_start_time = time.monotonic()
        self.query_one("#progress-clip-name", Label).update(clip_name)
        self.query_one("#encoding-progress-bar", ProgressBar).update(
            total=100, progress=0,
        )
        self.query_one("#progress-detail", Label).update("")
        self.query_one("#progress-overall", Label).update(
            f"Clip {clip_number} of {self._total_clips}"
        )

    def update_progress(self, percent: float) -> None:
        self.query_one("#encoding-progress-bar", ProgressBar).update(
            progress=percent,
        )
        eta_text = ""
        if percent > 0:
            elapsed_seconds = time.monotonic() - self._encoding_start_time
            remaining_seconds = elapsed_seconds * (100 - percent) / percent
            eta_text = f"ETA {self._format_duration(remaining_seconds)}"
        self.query_one("#progress-detail", Label).update(eta_text)

    def finish_clip(self) -> None:
        self.query_one("#encoding-progress-bar", ProgressBar).update(progress=100)
        elapsed_seconds = time.monotonic() - self._encoding_start_time
        self.query_one("#progress-detail", Label).update(
            f"Done in {self._format_duration(elapsed_seconds)}"
        )
