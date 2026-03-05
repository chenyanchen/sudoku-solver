"""Tests for screen monitor logic (cache, stability, tracking, rendering)."""

import numpy as np

from screen_monitor.frame_hasher import find_changed_regions, is_frame_changed, make_thumbnail
from screen_monitor.grid_tracker import (
    StabilityTracker,
    bbox_iou,
    grids_match,
    quantize_bbox,
    size_delta_ratio,
)
from screen_monitor.renderer import (
    build_render_key,
    compute_hint_positions,
    should_render,
)
from screen_monitor.types import LruCache, MonitorConfig


# ---------------------------------------------------------------------------
# FrameCache / frame_hasher
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LruCache
# ---------------------------------------------------------------------------


def test_lru_cache_basic_get_put():
    cache = LruCache(3)
    assert cache.get("a") is None

    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert len(cache) == 3
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_lru_cache_eviction():
    cache = LruCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts "a"

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_lru_cache_access_refreshes_order():
    cache = LruCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")     # refresh "a" — now "b" is oldest
    cache.put("c", 3)  # evicts "b"

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_lru_cache_max_size_one():
    cache = LruCache(1)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# bbox_iou
# ---------------------------------------------------------------------------


def test_bbox_iou_identical():
    box = (100, 100, 500, 500)
    assert abs(bbox_iou(box, box) - 1.0) < 1e-6


def test_bbox_iou_no_overlap():
    a = (0, 0, 100, 100)
    b = (200, 200, 300, 300)
    assert bbox_iou(a, b) == 0.0


def test_bbox_iou_partial_overlap():
    a = (0, 0, 100, 100)
    b = (50, 50, 150, 150)
    iou = bbox_iou(a, b)
    # intersection: 50x50=2500, union: 10000+10000-2500=17500
    assert abs(iou - 2500.0 / 17500.0) < 1e-6


def test_size_delta_ratio_identical():
    box = (0, 0, 100, 100)
    assert size_delta_ratio(box, box) == 0.0


def test_size_delta_ratio_different():
    a = (0, 0, 100, 100)
    b = (0, 0, 100, 200)
    ratio = size_delta_ratio(a, b)
    # areas: 10000, 20000; delta/max = 10000/20000 = 0.5
    assert abs(ratio - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# grids_match
# ---------------------------------------------------------------------------


def test_grids_match_by_signature_or_bbox():
    prev_bbox = (100, 100, 500, 500)
    cur_bbox_close = (110, 110, 510, 510)
    cur_bbox_far = (600, 100, 1000, 500)

    # Signature short-circuit.
    assert grids_match("same", prev_bbox, "same", cur_bbox_far, 0.85, 0.08) is True
    # IoU match.
    assert grids_match("old", prev_bbox, "new", cur_bbox_close, 0.80, 0.20) is True
    # No match.
    assert grids_match("old", prev_bbox, "new", cur_bbox_far, 0.80, 0.20) is False


# ---------------------------------------------------------------------------
# StabilityTracker
# ---------------------------------------------------------------------------


def test_stability_tracker_basic():
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


def test_stability_tracker_signature_change_resets():
    tracker = StabilityTracker()
    cfg = MonitorConfig(stable_frames=3)
    bbox = (100, 100, 500, 500)

    tracker.on_grid("sig-A", bbox, cfg)
    tracker.on_grid("sig-A", bbox, cfg)
    assert tracker.stable_count == 2

    # Different signature + far bbox → resets to 1.
    far_bbox = (800, 800, 1200, 1200)
    tracker.on_grid("sig-B", far_bbox, cfg)
    assert tracker.stable_count == 1


# ---------------------------------------------------------------------------
# render key / quantize
# ---------------------------------------------------------------------------


def test_render_key_quantization_and_should_render():
    key_1 = build_render_key("sig", (101, 103, 498, 502), dpr=2.0, quantize_step=4)
    key_2 = build_render_key("sig", (100, 104, 497, 503), dpr=2.0, quantize_step=4)
    key_3 = build_render_key("sig", (120, 120, 520, 520), dpr=2.0, quantize_step=4)

    assert key_1 == key_2
    assert should_render(key_1, key_2) is False
    assert should_render(key_1, key_3) is True


def test_quantize_bbox_returns_correct_type():
    result = quantize_bbox((101, 203, 305, 407), step=4)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert all(isinstance(v, int) for v in result)


# ---------------------------------------------------------------------------
# compute_hint_positions
# ---------------------------------------------------------------------------


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


def test_compute_hint_positions_with_retina_scale():
    """Retina 2x: capture coords are 2x logical → scale = 0.5."""
    corners = np.float32([[0, 0], [898, 0], [898, 898], [0, 898]])
    hints = [{"row": 4, "col": 4, "digit": 7}]

    output = compute_hint_positions(
        corners, hints,
        logical_offset=(100.0, 50.0),
        capture_to_logical_scale=(0.5, 0.5),
    )
    assert len(output) == 1
    # Center of cell (4,4) in warped space = (225, 225).
    # Transformed to 898x898 capture space = (450, 450).
    # Scaled to logical: 450 * 0.5 = 225 + offset.
    assert abs(output[0]["x"] - (100.0 + 225.0)) < 1.0
    assert abs(output[0]["y"] - (50.0 + 225.0)) < 1.0


def test_compute_hint_positions_empty_hints():
    corners = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
    assert compute_hint_positions(corners, []) == []


# ---------------------------------------------------------------------------
# P0: rescan_interval default
# ---------------------------------------------------------------------------


def test_monitor_config_rescan_interval_default():
    cfg = MonitorConfig()
    assert cfg.rescan_interval == 1.0


def test_monitor_config_rescan_interval_custom():
    cfg = MonitorConfig(rescan_interval=0.5)
    assert cfg.rescan_interval == 0.5


# ---------------------------------------------------------------------------
# P3: find_changed_regions
# ---------------------------------------------------------------------------


def test_find_changed_regions_identical_frames():
    thumb = np.zeros((90, 160), dtype=np.uint8)
    regions = find_changed_regions(thumb, thumb)
    assert regions == []


def test_find_changed_regions_detects_block():
    prev = np.zeros((90, 160), dtype=np.uint8)
    curr = prev.copy()
    # Paint a bright block in the center.
    curr[30:60, 60:100] = 255
    regions = find_changed_regions(prev, curr)
    assert len(regions) >= 1
    # The detected region should overlap with the changed area.
    x, y, w, h = regions[0]
    assert 0.2 < x < 0.7
    assert 0.2 < y < 0.8
    assert w > 0.05
    assert h > 0.05


def test_find_changed_regions_ignores_tiny_noise():
    prev = np.zeros((90, 160), dtype=np.uint8)
    curr = prev.copy()
    # Single pixel change — should be below min_area_frac.
    curr[45, 80] = 255
    regions = find_changed_regions(prev, curr, min_area_frac=0.01)
    assert regions == []
