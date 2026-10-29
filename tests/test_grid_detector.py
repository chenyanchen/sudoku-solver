"""Tests for snapping a rough quad to the 9x9 grid-line lattice."""

import cv2
import numpy as np

from backend.cv.grid_detector import snap_to_lattice

CELL, ORIGIN = 40, 100  # grid spans 100..460 px
TRUTH = np.float32([[100, 100], [460, 100], [460, 460], [100, 460]])
TOLERANCE = CELL / 4  # still read correctly by OCR


def _grid_image() -> np.ndarray:
    img = np.full((560, 560, 3), 235, np.uint8)
    for i in range(10):
        p = ORIGIN + i * CELL
        thick = 3 if i % 3 == 0 else 1
        cv2.line(img, (p, ORIGIN), (p, ORIGIN + 9 * CELL), (40, 40, 40), thick)
        cv2.line(img, (ORIGIN, p), (ORIGIN + 9 * CELL, p), (40, 40, 40), thick)
    return img


def _block(col: float, row: float, cells: float) -> np.ndarray:
    x, y, s = ORIGIN + col * CELL, ORIGIN + row * CELL, cells * CELL
    return np.float32([[x, y], [x + s, y], [x + s, y + s], [x, y + s]])


def test_extends_a_rough_box_to_the_whole_grid():
    # Corner errors of 2.5% of the box are amplified 3x when extrapolating.
    rough = _block(3, 3, 3) + np.float32([[3, -2], [-2, 2], [2, 3], [-3, -2]])
    result = snap_to_lattice(_grid_image(), rough, cells=3)
    assert result is not None
    assert np.abs(result[1] - TRUTH).max() < TOLERANCE


def test_shrinks_a_frame_to_the_grid():
    frame = _block(-0.6, -0.6, 10.2)
    result = snap_to_lattice(_grid_image(), frame, cells=9)
    assert result is not None
    assert np.abs(result[1] - TRUTH).max() < TOLERANCE


def test_rejects_a_3x3_block_off_the_box_boundaries():
    assert snap_to_lattice(_grid_image(), _block(1, 1, 3), cells=3) is None


def _tiles_image(height: int) -> np.ndarray:
    """Dark tiles on a light board (a game's grid has no drawn lines), with a
    panel one cell above and left of it: an edge spaced like a grid line."""
    img = np.full((560, 560, 3), 200, np.uint8)
    for k in range(-1, 9):
        a = ORIGIN + k * CELL + 3
        cv2.rectangle(img, (ORIGIN + 3, a), (ORIGIN + 9 * CELL - 3, a + CELL - 6), (60, 60, 60), -1)
        cv2.rectangle(img, (a, ORIGIN + 3), (a + CELL - 6, ORIGIN + 9 * CELL - 3), (60, 60, 60), -1)
    for row in range(9):
        for col in range(9):
            x, y = ORIGIN + col * CELL, ORIGIN + row * CELL
            cv2.rectangle(img, (x, y), (x + CELL, y + CELL), (200, 200, 200), 6)
    return img[:height]


def test_extends_a_cropped_grid_past_the_image():
    """A screenshot cut through the last row: its bottom line is past the image."""
    image = _tiles_image(height=450)
    rough = np.float32([[103, 104], [456, 98], [468, 449], [98, 449]])  # corners stop at the edge
    result = snap_to_lattice(image, rough, cells=9)
    assert result is not None
    assert np.abs(result[1] - TRUTH).max() < TOLERANCE
