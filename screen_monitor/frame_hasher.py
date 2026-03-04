"""Frame-level change detection using downsampled thumbnails."""

from __future__ import annotations

import hashlib
from typing import Optional

import cv2
import numpy as np


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
