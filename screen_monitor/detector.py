"""Grid detection with progressive ratio scanning and CLAHE retry."""

from __future__ import annotations

import cv2
import numpy as np

from backend.cv.grid_detector import find_grids_with_corners

from .grid_tracker import bbox_from_corners


def detect_candidates_with_corners(
    frame_bgr: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    return _detect_candidates_raw(frame_bgr, max_grids=8)


def _detect_candidates_raw(
    frame_bgr: np.ndarray,
    max_grids: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    ratios = (0.08, 0.04, 0.02)
    scan_limit = max(1, int(max_grids))

    for ratio in ratios:
        candidates = find_grids_with_corners(
            frame_bgr, max_grids=scan_limit, min_area_ratio=ratio,
        )
        if candidates:
            return _dedupe_candidates(candidates, max_candidates=scan_limit)

    # CLAHE-enhanced retry.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    for ratio in ratios:
        candidates = find_grids_with_corners(
            enhanced, max_grids=scan_limit, min_area_ratio=ratio,
        )
        if candidates:
            return _dedupe_candidates(candidates, max_candidates=scan_limit)

    return []


def _dedupe_candidates(
    candidates: list[tuple[np.ndarray, np.ndarray]],
    max_candidates: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for warped, corners in candidates:
        bbox = bbox_from_corners(corners)
        key = tuple(int(round(v / 12.0) * 12) for v in bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append((warped, corners))
        if len(unique) >= max_candidates:
            break

    return unique
