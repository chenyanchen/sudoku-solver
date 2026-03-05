"""Frame-level change detection using downsampled thumbnails."""

from __future__ import annotations

import hashlib
from typing import Optional

import cv2
import numpy as np


# Pre-allocated structuring element for find_changed_regions().
_MORPH_KERNEL_5x5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def make_thumbnail(
    frame_bgr: np.ndarray,
    size: tuple[int, int] = (160, 90),
) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def mean_abs_diff(img_a: np.ndarray, img_b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(img_a, img_b)))


def is_frame_changed(
    previous_thumb: Optional[np.ndarray],
    current_thumb: np.ndarray,
    threshold: float,
) -> bool:
    if previous_thumb is None:
        return True
    return mean_abs_diff(previous_thumb, current_thumb) > float(threshold)


def thumbnail_hash(thumb_gray: np.ndarray) -> str:
    return hashlib.sha1(thumb_gray.tobytes()).hexdigest()


def find_changed_regions(
    prev_thumb: np.ndarray,
    curr_thumb: np.ndarray,
    threshold: float = 20.0,
    min_area_frac: float = 0.005,
) -> list[tuple[float, float, float, float]]:
    """Find changed regions between two thumbnails.

    Returns fractional bounding boxes (x_frac, y_frac, w_frac, h_frac)
    normalized to [0, 1] relative to thumbnail dimensions.
    """
    diff = cv2.absdiff(prev_thumb, curr_thumb)
    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, _MORPH_KERNEL_5x5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    h, w = prev_thumb.shape[:2]
    total_area = float(h * w)
    regions: list[tuple[float, float, float, float]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < total_area * min_area_frac:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        regions.append((bx / w, by / h, bw / w, bh / h))

    return regions
