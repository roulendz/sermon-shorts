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
    discovered_audio_tracks: dict[str, Path] = field(default_factory=dict)
    output_directory: Path = Path("./output")

    # ── Set after WhisperX transcription ─────────────────────────────────────
    subtitle_file_path: Optional[Path] = None
    whisperx_version: Optional[int] = None

    # ── Set after Manus segment selection ────────────────────────────────────
    selected_segments: list[VideoSegment] = field(default_factory=list)

    # ── Auto flags — loaded from .env, skips human confirmation steps ────────
    auto_transcription: bool = False
    auto_segment_selection: bool = False
    auto_segment_review: bool = False
    auto_video_cutting: bool = False

    def has_discovered_audio_tracks(self) -> bool:
        return len(self.discovered_audio_tracks) > 0

    def is_ready_for_transcription(self) -> bool:
        return self.video_file_path is not None and self.has_discovered_audio_tracks()

    def is_ready_for_segment_selection(self) -> bool:
        return self.subtitle_file_path is not None and self.subtitle_file_path.exists()

    def is_ready_for_cutting(self) -> bool:
        return (
            self.video_file_path is not None
            and len(self.selected_segments) > 0
        )
