"""
tests/conftest.py
Shared pytest fixtures used across all test modules.
"""

import pytest
import srt
from datetime import timedelta
from models.video_segment import VideoSegment


def make_timedelta(hours: int = 0, minutes: int = 0, seconds: int = 0, milliseconds: int = 0) -> timedelta:
    return timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)


@pytest.fixture
def simple_subtitle_list() -> list[srt.Subtitle]:
    return [
        srt.Subtitle(index=1, start=make_timedelta(0, 1, 0), end=make_timedelta(0, 1, 5), content="Hello world."),
        srt.Subtitle(index=2, start=make_timedelta(0, 1, 6), end=make_timedelta(0, 1, 10), content="This is a sermon."),
        srt.Subtitle(index=3, start=make_timedelta(0, 5, 0), end=make_timedelta(0, 5, 8), content="God is good."),
        srt.Subtitle(index=4, start=make_timedelta(0, 5, 9), end=make_timedelta(0, 5, 20), content="All the time."),
        srt.Subtitle(index=5, start=make_timedelta(0, 10, 0), end=make_timedelta(0, 10, 15), content="Final words here."),
    ]


@pytest.fixture
def simple_video_segment() -> VideoSegment:
    return VideoSegment(
        index=1,
        start_time=make_timedelta(0, 5, 0),
        end_time=make_timedelta(0, 5, 27),
        transcript_text="God is good. All the time.",
        selection_reason="Strong emotional moment",
    )
