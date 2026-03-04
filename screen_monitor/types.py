"""Shared configuration and data structures."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MonitorConfig:
    model_path: Optional[str] = None
    monitor_index: int = 0
    idle_fps: float = 2.0
    active_fps: float = 6.0
    active_seconds: float = 2.0
    stable_frames: int = 2
    lost_frames: int = 3
    frame_change_threshold: float = 1.8
    bbox_iou_threshold: float = 0.85
    grid_size_delta_ratio: float = 0.08
    min_givens: int = 17
    min_bbox_area_ratio: float = 0.03
    overlay_hold_seconds: float = 2.0
    solve_cache_max: int = 20
    detect_cache_max: int = 64
    quantize_step: int = 4
    debug: bool = False


class LruCache:
    """Small LRU cache backed by OrderedDict."""

    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)
