"""Lighting gradients must not turn empty photo cells into digits."""

import cv2
import numpy as np

from backend.ocr.pipeline import prepare_grid_for_ocr
from backend.cv.cell_extractor import clean_cell, extract_cells


def test_photo_lighting_gradient_stays_blank_but_preserves_digit():
    gray = np.tile(np.linspace(120, 240, 450, dtype=np.uint8), (450, 1))
    cv2.putText(gray, "5", (210, 239), cv2.FONT_HERSHEY_SIMPLEX, 1, 30, 2)
    board = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cells = [clean_cell(c) for c in extract_cells(prepare_grid_for_ocr(board))]
    assert max(np.mean(c < 128) for i, c in enumerate(cells) if i != 40) < 0.01
    assert 0.03 < np.mean(cells[40] < 128) < 0.3
