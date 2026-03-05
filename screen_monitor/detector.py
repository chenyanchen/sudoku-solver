"""Grid detection for screen captures: direct scan + content-region upscale."""

from __future__ import annotations

import cv2
import numpy as np

from backend.cv.grid_detector import find_grids_with_corners

from .grid_tracker import bbox_from_corners

# Minimum area ratio for grid candidates.  On Retina displays the sudoku
# grid can be < 1% of the full screen, so we use a low floor.
_MIN_AREA_RATIO = 0.005

# Content-region upscale target width.  Thin-bordered grids (e.g. phone
# app screenshots) are too small after screen capture; upscaling the
# content region to this width restores enough detail for detection.
_UPSCALE_TARGET_W = 1200

# Reusable CLAHE instance (avoid per-frame allocation).
_CLAHE = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))


def detect_candidates_with_corners(
    frame_bgr: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Detect sudoku grid candidates in a screen capture frame.

    Uses a two-stage strategy:
    1. Direct detection on the full frame (fast path for large grids).
    2. Content-region upscale: find the non-background area, upscale it,
       and re-detect.  This handles thin-bordered grids displayed inside
       image viewers or browser windows on high-resolution screens.
    """
    max_grids = 8

    # --- Stage 1: direct detection on full frame ---
    candidates = find_grids_with_corners(
        frame_bgr, max_grids=max_grids, min_area_ratio=_MIN_AREA_RATIO,
    )
    if candidates:
        deduped = _dedupe_candidates(candidates, max_candidates=max_grids)
        # Accept if at least one candidate is roughly square (likely grid).
        if any(_is_roughly_square(corners) for _, corners in deduped):
            return deduped

    # --- Stage 2: content-region upscale ---
    regions = _find_content_regions(frame_bgr, min_area_frac=0.005, max_regions=3)
    all_candidates: list[tuple[np.ndarray, np.ndarray]] = []

    for rx, ry, rw, rh in regions:
        crop = frame_bgr[ry : ry + rh, rx : rx + rw]

        # Upscale if the content region is small.
        scale_up = max(1.0, _UPSCALE_TARGET_W / max(crop.shape[1], 1))
        if scale_up > 1.0:
            upscaled = cv2.resize(
                crop, None, fx=scale_up, fy=scale_up,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            upscaled = crop

        grids = find_grids_with_corners(
            upscaled, max_grids=max_grids, min_area_ratio=0.01,
        )

        for warped, corners in grids:
            # Map corners back to original frame coordinates.
            corners = corners / scale_up
            corners[:, 0] += rx
            corners[:, 1] += ry
            all_candidates.append((warped, corners))

        # Early exit: stop after the first region that yields candidates.
        if all_candidates:
            break

    if all_candidates:
        return _dedupe_candidates(all_candidates, max_candidates=max_grids)

    # --- Stage 3: CLAHE enhanced retry on full frame ---
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.cvtColor(_CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)
    candidates = find_grids_with_corners(
        enhanced, max_grids=max_grids, min_area_ratio=_MIN_AREA_RATIO,
    )
    if candidates:
        return _dedupe_candidates(candidates, max_candidates=max_grids)

    return []


def detect_candidates_in_regions(
    frame_bgr: np.ndarray,
    regions: list[tuple[float, float, float, float]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Detect sudoku grid candidates within specific ROI regions.

    Args:
        frame_bgr: Full frame in BGR format.
        regions: Fractional bounding boxes (x_frac, y_frac, w_frac, h_frac).

    Returns:
        List of (warped, corners) with corners mapped back to full-frame coords.
    """
    h, w = frame_bgr.shape[:2]
    all_candidates: list[tuple[np.ndarray, np.ndarray]] = []

    for x_frac, y_frac, w_frac, h_frac in regions:
        # Convert fractional coords to pixel coords with padding.
        pad_frac = 0.05
        x1 = max(0, int((x_frac - pad_frac) * w))
        y1 = max(0, int((y_frac - pad_frac) * h))
        x2 = min(w, int((x_frac + w_frac + pad_frac) * w))
        y2 = min(h, int((y_frac + h_frac + pad_frac) * h))

        if x2 - x1 < 50 or y2 - y1 < 50:
            continue

        crop = frame_bgr[y1:y2, x1:x2]

        # Upscale small crops for better detection.
        scale_up = max(1.0, _UPSCALE_TARGET_W / max(crop.shape[1], 1))
        if scale_up > 1.0:
            upscaled = cv2.resize(
                crop, None, fx=scale_up, fy=scale_up,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            upscaled = crop

        grids = find_grids_with_corners(
            upscaled, max_grids=4, min_area_ratio=0.01,
        )

        for warped, corners in grids:
            # Map corners back to full-frame coordinates.
            corners = corners / scale_up
            corners[:, 0] += x1
            corners[:, 1] += y1
            all_candidates.append((warped, corners))

    if all_candidates:
        return _dedupe_candidates(all_candidates, max_candidates=8)
    return []


def _is_roughly_square(corners: np.ndarray, min_ratio: float = 0.5) -> bool:
    """Check if corners form a roughly square quadrilateral."""
    w = float(np.linalg.norm(corners[1] - corners[0]))
    h = float(np.linalg.norm(corners[3] - corners[0]))
    if max(w, h) < 1.0:
        return False
    return min(w, h) / max(w, h) >= min_ratio


def _find_content_regions(
    frame_bgr: np.ndarray,
    min_area_frac: float = 0.005,
    max_regions: int = 3,
) -> list[tuple[int, int, int, int]]:
    """Find non-background rectangular regions in a screen capture.

    Returns list of (x, y, w, h) bounding boxes.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Estimate background colour from screen corners.
    corner_vals = [
        gray[0, 0], gray[0, w - 1], gray[h - 1, 0], gray[h - 1, w - 1],
    ]
    bg_val = int(np.median(corner_vals))

    diff = cv2.absdiff(gray, np.full_like(gray, bg_val))
    _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.dilate(mask, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    image_area = float(h * w)
    regions: list[tuple[float, tuple[int, int, int, int]]] = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < image_area * min_area_frac:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        # Add small padding.
        pad = max(5, int(max(bw, bh) * 0.02))
        bx = max(0, bx - pad)
        by = max(0, by - pad)
        bw = min(w - bx, bw + 2 * pad)
        bh = min(h - by, bh + 2 * pad)
        regions.append((area, (bx, by, bw, bh)))

    regions.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in regions[:max_regions]]


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
