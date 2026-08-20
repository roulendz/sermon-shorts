"""Tests for pipeline/timeline_receipt.py."""

import json

import pytest

from pipeline.timeline_receipt import (
    build_base_stage,
    build_portrait_stage,
    build_receipt_file_path,
    build_trim_stage,
    load_timeline_receipt,
    map_subtitle_time_to_final,
    save_timeline_receipt,
)


# ── stage builders ───────────────────────────────────────────────────────────

def test_build_base_stage_computes_head_padding():
    stage = build_base_stage(100.0, 184.9, 86.25)
    assert stage["headPaddingSec"] == pytest.approx(1.35)
    assert stage["requestedStartSec"] == 100.0
    assert stage["requestedEndSec"] == 184.9


def test_build_base_stage_clamps_negative_padding_to_zero():
    stage = build_base_stage(0.0, 60.0, 59.8)
    assert stage["headPaddingSec"] == 0.0


def test_build_trim_stage_rounds_segments():
    stage = build_trim_stage([(0.0, 10.55555), (12.0, 20.0)])
    assert stage["keptSegments"] == [[0.0, 10.556], [12.0, 20.0]]


# ── mapping ──────────────────────────────────────────────────────────────────

def make_receipt(head_padding=1.3, speed=1.3, kept_segments=None):
    receipt = {
        "version": 1,
        "clipBase": "clip",
        "base": {
            "requestedStartSec": 100.0,
            "requestedEndSec": 185.0,
            "mp4DurationSec": 85.0 + head_padding,
            "headPaddingSec": head_padding,
        },
        "portrait": {"sourceDurationSec": 85.0 + head_padding, "speedMultiplier": speed},
    }
    if kept_segments is not None:
        receipt["trim"] = {"keptSegments": kept_segments}
    return receipt


def test_map_applies_head_padding_and_speed():
    receipt = make_receipt(head_padding=1.3, speed=1.3)
    assert map_subtitle_time_to_final(0.0, receipt) == pytest.approx(1.0)
    assert map_subtitle_time_to_final(11.7, receipt) == pytest.approx(10.0)


def test_map_without_portrait_stage_applies_padding_only():
    receipt = make_receipt(head_padding=0.5)
    del receipt["portrait"]
    assert map_subtitle_time_to_final(10.0, receipt) == pytest.approx(10.5)


def test_map_through_trim_inside_first_kept_segment():
    receipt = make_receipt(head_padding=0.0, speed=1.0,
                           kept_segments=[[0.0, 10.0], [15.0, 25.0]])
    assert map_subtitle_time_to_final(4.0, receipt) == pytest.approx(4.0)


def test_map_through_trim_inside_later_kept_segment():
    receipt = make_receipt(head_padding=0.0, speed=1.0,
                           kept_segments=[[0.0, 10.0], [15.0, 25.0]])
    # 5s gap removed: portrait 20.0 -> trimmed 10 + (20-15) = 15
    assert map_subtitle_time_to_final(20.0, receipt) == pytest.approx(15.0)


def test_map_returns_none_inside_removed_gap():
    receipt = make_receipt(head_padding=0.0, speed=1.0,
                           kept_segments=[[0.0, 10.0], [15.0, 25.0]])
    assert map_subtitle_time_to_final(12.0, receipt) is None


def test_map_returns_none_past_last_kept_segment():
    receipt = make_receipt(head_padding=0.0, speed=1.0,
                           kept_segments=[[0.0, 10.0]])
    assert map_subtitle_time_to_final(11.0, receipt) is None


def test_map_full_chain_padding_speed_trim():
    receipt = make_receipt(head_padding=1.3, speed=1.3,
                           kept_segments=[[0.0, 5.0], [8.0, 60.0]])
    # subtitle 11.7 -> mp4 13.0 -> portrait 10.0 -> trimmed 5 + (10-8) = 7
    assert map_subtitle_time_to_final(11.7, receipt) == pytest.approx(7.0)


# ── persistence ──────────────────────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    receipt = make_receipt()
    saved_path = save_timeline_receipt(tmp_path, receipt)
    assert saved_path == build_receipt_file_path(tmp_path, "clip")
    loaded = load_timeline_receipt(tmp_path, "clip")
    assert loaded["base"]["headPaddingSec"] == receipt["base"]["headPaddingSec"]
    assert "updatedUtc" in loaded


def test_load_missing_receipt_returns_skeleton(tmp_path):
    receipt = load_timeline_receipt(tmp_path, "nope")
    assert receipt == {"version": 1, "clipBase": "nope"}


def test_saved_receipt_is_valid_json_with_unicode(tmp_path):
    receipt = make_receipt()
    receipt["clipBase"] = "2026-07-19 Augšāmcelšanās [1m42s]"
    save_timeline_receipt(tmp_path, receipt)
    raw = build_receipt_file_path(tmp_path, receipt["clipBase"]).read_text(encoding="utf-8")
    assert "Augšāmcelšanās" in raw
    assert json.loads(raw)["version"] == 1
