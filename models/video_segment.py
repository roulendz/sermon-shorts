"""
models/video_segment.py

Represents a single selected video segment with its time boundaries,
transcript text, AI-generated selection reason, viral scoring, and hook overlay.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class VideoSegment:
    index: int
    start_time: timedelta
    end_time: timedelta
    transcript_text: str
    selection_reason: str

    # Viral scoring — populated from Manus AI response
    hook_power_score: int = 0
    emotional_impact_score: int = 0
    discussion_potential_score: int = 0
    shareability_score: int = 0
    overall_viral_score: float = 0.0

    # AI-generated title for the clip
    suggested_title: str = ""

    # Short text overlay for the opening of the clip — for Phase 2 burn-in
    viral_hook_text_overlay: str = ""

    def __post_init__(self) -> None:
        if self.start_time < timedelta(0):
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be greater than start_time ({self.start_time})"
            )
        if not self.transcript_text.strip():
            raise ValueError("transcript_text must not be empty")

    @property
    def duration(self) -> timedelta:
        return self.end_time - self.start_time

    @property
    def duration_seconds(self) -> float:
        return self.duration.total_seconds()

    @property
    def start_seconds(self) -> float:
        return self.start_time.total_seconds()

    @property
    def end_seconds(self) -> float:
        return self.end_time.total_seconds()

    def as_ffmpeg_timestamp(self, time: timedelta) -> str:
        """Format a timedelta as HH:MM:SS.mmm for FFmpeg -ss / -to arguments."""
        total_seconds = int(time.total_seconds())
        milliseconds = int(time.microseconds / 1000)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @property
    def ffmpeg_start_timestamp(self) -> str:
        return self.as_ffmpeg_timestamp(self.start_time)

    @property
    def ffmpeg_end_timestamp(self) -> str:
        return self.as_ffmpeg_timestamp(self.end_time)

    def __repr__(self) -> str:
        return (
            f"VideoSegment(index={self.index}, "
            f"start={self.ffmpeg_start_timestamp}, "
            f"end={self.ffmpeg_end_timestamp}, "
            f"duration={self.duration_seconds:.1f}s, "
            f"viral_score={self.overall_viral_score:.1f})"
        )
