"""
models/pipeline_state.py

Single dataclass that carries all state through the pipeline.
Passed from screen to screen in the TUI and step to step in the pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from models.video_segment import VideoSegment


@dataclass
class PipelineState:
    # ── File paths set during file selection screen ──────────────────────────
    video_file_path: Optional[Path] = None
    audio_file_path: Optional[Path] = None
    output_directory: Path = Path("./output")

    # ── Set during audio alignment screen ────────────────────────────────────
    # Positive value means audio starts this many seconds INTO the video.
    # e.g. 45.3 means the audio file begins at 00:00:45.3 of the video.
    audio_to_video_offset_seconds: float = 0.0

    # ── Set after WhisperX transcription ─────────────────────────────────────
    subtitle_file_path: Optional[Path] = None

    # ── Set after Manus segment selection ────────────────────────────────────
    selected_segments: list[VideoSegment] = field(default_factory=list)

    # ── Auto flags — loaded from .env, skips human confirmation steps ────────
    auto_alignment: bool = False
    auto_transcription: bool = False
    auto_segment_selection: bool = False
    auto_segment_review: bool = False
    auto_video_cutting: bool = False

    def is_ready_for_alignment(self) -> bool:
        return self.video_file_path is not None and self.audio_file_path is not None

    def is_ready_for_transcription(self) -> bool:
        return self.audio_file_path is not None

    def is_ready_for_segment_selection(self) -> bool:
        return self.subtitle_file_path is not None and self.subtitle_file_path.exists()

    def is_ready_for_cutting(self) -> bool:
        return (
            self.video_file_path is not None
            and len(self.selected_segments) > 0
        )
