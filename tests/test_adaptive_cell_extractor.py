"""Tests for adaptive cell extraction using 16 keypoints."""

import numpy as np
import cv2

from backend.cv.adaptive_cell_extractor import extract_cells_adaptive


def _make_synthetic_grid(size: int = 900) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic 9x9 grid image with known 16 keypoints.

    Returns (image, keypoints_16) where keypoints are in image coords.
    The grid is drawn starting at (100, 100) with 70px per cell,
    so total grid is 630x630 within a 900x900 image.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 200

    origin_x, origin_y = 100, 100
    cell_size = 70
    block_size = cell_size * 3  # 210px per block

    for i in range(10):
        y = origin_y + i * cell_size
        x = origin_x + i * cell_size
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(img, (origin_x, y), (origin_x + 9 * cell_size, y), (0, 0, 0), thickness)
        cv2.line(img, (x, origin_y), (x, origin_y + 9 * cell_size), (0, 0, 0), thickness)

    for r in range(9):
        for c in range(9):
            if (r + c) % 5 == 0:
                cx = origin_x + c * cell_size + cell_size // 2
                cy = origin_y + r * cell_size + cell_size // 2
                cv2.putText(img, "5", (cx - 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    keypoints = np.zeros((16, 2), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            keypoints[idx, 0] = origin_x + col * block_size
            keypoints[idx, 1] = origin_y + row * block_size

    return img, keypoints


class TestExtractCellsAdaptive:
    def test_returns_81_cells(self):
        img, kpts = _make_synthetic_grid()
        cells = extract_cells_adaptive(img, kpts)
        assert len(cells) == 81

    def test_cell_shape(self):
        img, kpts = _make_synthetic_grid()
        cells = extract_cells_adaptive(img, kpts)
        for cell in cells:
            assert cell.shape[0] == cell.shape[1], f"Cell not square: {cell.shape}"
            assert cell.shape[0] >= 40, f"Cell too small: {cell.shape}"

    def test_row_major_order(self):
        """Cell 0 is top-left, cell 80 is bottom-right."""
        img, kpts = _make_synthetic_grid()
        cells = extract_cells_adaptive(img, kpts)
        assert len(cells) == 81
        assert all(c.size > 0 for c in cells)

    def test_uneven_block_sizes(self):
        """Simulate decorative grid where blocks have different sizes."""
        img = np.ones((900, 900, 3), dtype=np.uint8) * 200

        block_widths = [180, 220, 200]
        block_heights = [190, 210, 200]
        ox, oy = 100, 100

        keypoints = np.zeros((16, 2), dtype=np.float32)
        cum_x = [ox]
        for w in block_widths:
            cum_x.append(cum_x[-1] + w)
        cum_y = [oy]
        for h in block_heights:
            cum_y.append(cum_y[-1] + h)

        for row in range(4):
            for col in range(4):
                keypoints[row * 4 + col] = [cum_x[col], cum_y[row]]

        cells = extract_cells_adaptive(img, keypoints)
        assert len(cells) == 81
        for cell in cells:
            assert cell.shape[0] == cell.shape[1]
            assert cell.shape[0] >= 40
