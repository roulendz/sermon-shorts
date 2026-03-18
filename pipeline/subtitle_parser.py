"""
pipeline/subtitle_parser.py

Parses SRT subtitle files, shifts all timestamps by a given offset,
and slices subtitle entries that fall within a given time window.
No external dependencies beyond the stdlib `srt` package.
"""

from __future__ import annotations
from datetime import timedelta
from pathlib import Path
from typing import Sequence

import srt


def load_subtitle_file(subtitle_file_path: Path) -> list[srt.Subtitle]:
    raw_text = subtitle_file_path.read_text(encoding="utf-8")
    return parse_subtitle_text(raw_text)


def parse_subtitle_text(raw_srt_text: str) -> list[srt.Subtitle]:
    return list(srt.parse(raw_srt_text))


def shift_subtitle_timestamps(
    subtitles: list[srt.Subtitle],
    offset: timedelta,
) -> list[srt.Subtitle]:
    """
    Shift every subtitle's start and end by the given offset.
    Used to align audio-derived timestamps to the video timeline.
    A positive offset moves subtitles later in time.
    """
    shifted = []
    for subtitle in subtitles:
        shifted_start = subtitle.start + offset
        shifted_end = subtitle.end + offset

        # Drop subtitles that would become negative after shifting
        if shifted_start < timedelta(0):
            continue

        shifted.append(
            srt.Subtitle(
                index=subtitle.index,
                start=shifted_start,
                end=shifted_end,
                content=subtitle.content,
            )
        )
    return shifted


def slice_subtitles_within_window(
    subtitles: Sequence[srt.Subtitle],
    window_start: timedelta,
    window_end: timedelta,
) -> list[srt.Subtitle]:
    """
    Return only subtitles that fall within [window_start, window_end].
    Re-indexes from 1 and resets timestamps relative to window_start
    so the output SRT syncs with a video clip cut at window_start.
    """
    subtitles_in_window = [
        subtitle for subtitle in subtitles
        if subtitle.start >= window_start and subtitle.end <= window_end
    ]

    return reindex_and_rebase_subtitles(subtitles_in_window, window_start)


def reindex_and_rebase_subtitles(
    subtitles: Sequence[srt.Subtitle],
    time_base: timedelta,
) -> list[srt.Subtitle]:
    """
    Re-number subtitles starting from 1 and subtract time_base from
    all timestamps so they start from zero. Used when creating SRT
    files for extracted video clips.
    """
    return [
        srt.Subtitle(
            index=new_index,
            start=subtitle.start - time_base,
            end=subtitle.end - time_base,
            content=subtitle.content,
        )
        for new_index, subtitle in enumerate(subtitles, start=1)
    ]


def extract_plain_text_from_subtitles(subtitles: Sequence[srt.Subtitle]) -> str:
    """
    Concatenate all subtitle content into a single readable string.
    Used for displaying segment transcript text in the review screen.
    """
    return " ".join(subtitle.content.strip() for subtitle in subtitles)


def serialize_subtitles_to_srt_text(subtitles: Sequence[srt.Subtitle]) -> str:
    return srt.compose(subtitles)


def save_subtitles_to_file(
    subtitles: Sequence[srt.Subtitle],
    output_file_path: Path,
) -> None:
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_path.write_text(
        serialize_subtitles_to_srt_text(subtitles),
        encoding="utf-8",
    )
