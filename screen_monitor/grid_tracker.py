"""Grid tracking: IoU matching, stability counting, and signature helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .types import MonitorConfig


@dataclass
class StabilityTracker:
    stable_count: int = 0
    lost_count: int = 0
    last_signature: Optional[str] = None
    last_bbox: Optional[tuple[int, int, int, int]] = None

    def on_no_grid(self) -> int:
        self.lost_count += 1
        self.stable_count = 0
        return self.lost_count

    def on_grid(
        self,
        signature: str,
        bbox: tuple[int, int, int, int],
        cfg: MonitorConfig,
    ) -> int:
        if grids_match(
            self.last_signature,
            self.last_bbox,
            signature,
            bbox,
            cfg.bbox_iou_threshold,
            cfg.grid_size_delta_ratio,
        ):
            self.stable_count += 1
        else:
            self.stable_count = 1
        self.lost_count = 0
        self.last_signature = signature
        self.last_bbox = bbox
        return self.stable_count


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------


def bbox_from_corners(corners: np.ndarray) -> tuple[int, int, int, int]:
    xs = corners[:, 0]
    ys = corners[:, 1]
    return int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))


def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(area_a + area_b - inter_area)


def size_delta_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return abs(area_a - area_b) / float(max(area_a, area_b))


def grids_match(
    previous_signature: Optional[str],
    previous_bbox: Optional[tuple[int, int, int, int]],
    current_signature: str,
    current_bbox: tuple[int, int, int, int],
    iou_threshold: float,
    area_delta_ratio_threshold: float,
) -> bool:
    if previous_signature == current_signature and previous_signature is not None:
        return True
    if previous_bbox is None:
        return False
    if bbox_iou(previous_bbox, current_bbox) < iou_threshold:
        return False
    return size_delta_ratio(previous_bbox, current_bbox) <= area_delta_ratio_threshold


def quantize_bbox(
    bbox: tuple[int, int, int, int],
    step: int,
) -> tuple[int, int, int, int]:
    q = max(1, int(step))
    x1, y1, x2, y2 = bbox
    return (
        int(round(x1 / q) * q),
        int(round(y1 / q) * q),
        int(round(x2 / q) * q),
        int(round(y2 / q) * q),
    )


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------


def grid_signature(warped_bgr: np.ndarray) -> str:
    """Colour-invariant visual hash of a warped grid.

    Converts to binary (digits/lines vs. background) so that tinted
    backgrounds (NYTimes selection highlight, 3x3 block shading) do not
    change the hash.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(small, 128, 255, cv2.THRESH_BINARY)
    return hashlib.sha1(binary.tobytes()).hexdigest()


def puzzle_signature(grid: list[list[int]]) -> str:
    flat = "".join(str(int(cell)) for row in grid for cell in row)
    return hashlib.sha1(flat.encode("utf-8")).hexdigest()
