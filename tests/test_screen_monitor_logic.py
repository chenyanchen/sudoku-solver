"""Tests for screen monitor logic (cache keys, stability, and mapping)."""

import numpy as np

from scripts.screen_monitor import (
    MonitorConfig,
    StabilityTracker,
    build_render_key,
    compute_hint_positions,
    grids_match,
    is_frame_changed,
    make_thumbnail,
    should_render,
)


def test_frame_change_threshold_logic():
    frame_a = np.zeros((200, 300, 3), dtype=np.uint8)
    frame_b = np.zeros((200, 300, 3), dtype=np.uint8)
    thumb_a = make_thumbnail(frame_a)
    thumb_b = make_thumbnail(frame_b)

    assert is_frame_changed(None, thumb_a, threshold=1.8) is True
    assert is_frame_changed(thumb_a, thumb_b, threshold=1.8) is False

    frame_c = np.ones((200, 300, 3), dtype=np.uint8) * 255
    thumb_c = make_thumbnail(frame_c)
    assert is_frame_changed(thumb_a, thumb_c, threshold=1.8) is True


def test_grids_match_by_signature_or_bbox():
    prev_bbox = (100, 100, 500, 500)
    cur_bbox_close = (110, 110, 510, 510)
    cur_bbox_far = (600, 100, 1000, 500)

    assert (
        grids_match("same", prev_bbox, "same", cur_bbox_far, 0.85, 0.08) is True
    )  # Signature short-circuit
    assert grids_match("old", prev_bbox, "new", cur_bbox_close, 0.80, 0.20) is True
    assert grids_match("old", prev_bbox, "new", cur_bbox_far, 0.80, 0.20) is False


def test_stability_tracker_and_lost_counter():
    tracker = StabilityTracker()
    cfg = MonitorConfig(stable_frames=2)
    bbox = (100, 100, 500, 500)

    assert tracker.on_grid("sig-1", bbox, cfg) == 1
    assert tracker.on_grid("sig-1", bbox, cfg) == 2
    assert tracker.stable_count == 2
    assert tracker.lost_count == 0

    assert tracker.on_no_grid() == 1
    assert tracker.stable_count == 0
    assert tracker.on_no_grid() == 2


def test_render_key_quantization_and_should_render():
    key_1 = build_render_key("sig", (101, 103, 498, 502), dpr=2.0, quantize_step=4)
    key_2 = build_render_key("sig", (100, 104, 497, 503), dpr=2.0, quantize_step=4)
    key_3 = build_render_key("sig", (120, 120, 520, 520), dpr=2.0, quantize_step=4)

    assert key_1 == key_2
    assert should_render(key_1, key_2) is False
    assert should_render(key_1, key_3) is True


def test_compute_hint_positions_identity_mapping():
    corners = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
    hints = [{"row": 0, "col": 0, "digit": 5}, {"row": 8, "col": 8, "digit": 9}]

    output = compute_hint_positions(corners, hints)
    assert len(output) == 2
    assert output[0]["digit"] == 5
    assert abs(output[0]["x"] - 25.0) < 1e-3
    assert abs(output[0]["y"] - 25.0) < 1e-3
    assert output[1]["digit"] == 9
    assert abs(output[1]["x"] - 425.0) < 1e-3
    assert abs(output[1]["y"] - 425.0) < 1e-3
