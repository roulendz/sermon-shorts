"""Tests for pipeline/clip_description.py"""

from datetime import timedelta

from models.video_segment import VideoSegment
from pipeline.clip_description import (
    FIXED_DESCRIPTION_HEADER,
    build_clip_description_markdown,
    write_clip_description_file,
)


def _segment(social_description: str = "", suggested_title: str = "") -> VideoSegment:
    return VideoSegment(
        index=1,
        start_time=timedelta(seconds=10),
        end_time=timedelta(seconds=70),
        transcript_text="some transcript",
        selection_reason="reason",
        suggested_title=suggested_title,
        social_description=social_description,
    )


def test_markdown_contains_fixed_header():
    markdown = build_clip_description_markdown(_segment(social_description="Jautajums?\n#shorts"))
    assert FIXED_DESCRIPTION_HEADER in markdown


def test_markdown_contains_dynamic_body():
    markdown = build_clip_description_markdown(_segment(social_description="Kapec?\n#shorts"))
    assert "Kapec?" in markdown
    assert markdown.rstrip().endswith("#shorts")


def test_markdown_includes_title_heading_when_present():
    markdown = build_clip_description_markdown(_segment(social_description="x", suggested_title="My Title"))
    assert markdown.startswith("# My Title")


def test_markdown_falls_back_to_shorts_when_body_empty():
    markdown = build_clip_description_markdown(_segment(social_description=""))
    assert FIXED_DESCRIPTION_HEADER in markdown
    assert markdown.rstrip().endswith("#shorts")


def test_write_clip_description_file_uses_md_suffix(tmp_path):
    clip_video_path = tmp_path / "My Clip [57s].mp4"
    written_path = write_clip_description_file(
        _segment(social_description="Q?\n#shorts"), clip_video_path
    )
    assert written_path == tmp_path / "My Clip [57s].md"
    assert written_path.exists()
    assert FIXED_DESCRIPTION_HEADER in written_path.read_text(encoding="utf-8")
