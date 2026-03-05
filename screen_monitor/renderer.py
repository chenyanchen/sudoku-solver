"""Perspective back-projection and render-key helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .grid_tracker import quantize_bbox


def compute_hint_positions(
    corners: np.ndarray,
    hints: list[dict[str, Any]],
    logical_offset: tuple[float, float] = (0.0, 0.0),
    capture_to_logical_scale: tuple[float, float] = (1.0, 1.0),
) -> list[dict[str, Any]]:
    """Map each hint cell from warped-grid space back to screen logical points."""
    if not hints:
        return []

    dst = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
    matrix = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))

    source_points = []
    for item in hints:
        row = int(item["row"])
        col = int(item["col"])
        source_points.append([(col + 0.5) * 50.0, (row + 0.5) * 50.0])

    points = np.float32(source_points).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    x0 = float(logical_offset[0])
    y0 = float(logical_offset[1])
    sx = max(float(capture_to_logical_scale[0]), 1e-6)
    sy = max(float(capture_to_logical_scale[1]), 1e-6)

    output = []
    for i, item in enumerate(hints):
        output.append({
            "x": float(x0 + transformed[i, 0] * sx),
            "y": float(y0 + transformed[i, 1] * sy),
            "digit": int(item["digit"]),
        })
    return output


def compute_all_cell_positions(
    corners: np.ndarray,
    cell_predictions: list[dict[str, Any]],
    logical_offset: tuple[float, float] = (0.0, 0.0),
    capture_to_logical_scale: tuple[float, float] = (1.0, 1.0),
) -> list[dict[str, Any]]:
    """Map all 81 cell predictions to screen coordinates with confidence."""
    if not cell_predictions:
        return []

    dst = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
    matrix = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))

    source_points = []
    for pred in cell_predictions:
        row = int(pred["row"])
        col = int(pred["col"])
        source_points.append([(col + 0.5) * 50.0, (row + 0.5) * 50.0])

    points = np.float32(source_points).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    x0 = float(logical_offset[0])
    y0 = float(logical_offset[1])
    sx = max(float(capture_to_logical_scale[0]), 1e-6)
    sy = max(float(capture_to_logical_scale[1]), 1e-6)

    # Estimate cell size from grid extent.
    all_x = transformed[:, 0] * sx
    all_y = transformed[:, 1] * sy
    cell_size = max(8, int(min(all_x.ptp(), all_y.ptp()) / 9.0 * 0.85)) if len(transformed) > 1 else 20

    output = []
    for i, pred in enumerate(cell_predictions):
        output.append({
            "x": float(x0 + transformed[i, 0] * sx),
            "y": float(y0 + transformed[i, 1] * sy),
            "value": int(pred.get("value", 0)),
            "confidence": float(pred.get("confidence", 0.0)),
            "size": cell_size,
        })
    return output


def build_render_key(
    signature: str,
    bbox: tuple[int, int, int, int],
    dpr: float,
    quantize_step: int,
) -> tuple[str, tuple[int, int, int, int], float]:
    return (signature, quantize_bbox(bbox, quantize_step), round(float(dpr), 3))


def should_render(last_key: Any, new_key: Any) -> bool:
    return last_key != new_key
