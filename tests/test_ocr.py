"""Tests for OCR components."""

from pathlib import Path

import pytest
import numpy as np
import cv2

from backend.api.routes import detect_grid_image, prepare_grid_for_ocr
from backend.cv.cell_extractor import extract_cells
from backend.ocr.digit_reader import (
    DigitReader,
    recognize_digit,
    recognize_grid,
    preprocess_for_ocr,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "images"

EXPECTED_SAMPLE_GRIDS = {
    "sudoku_2.png": [
        [6, 0, 8, 0, 0, 3, 0, 2, 4],
        [4, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 8, 0],
        [8, 6, 0, 4, 7, 0, 0, 3, 0],
        [0, 7, 4, 1, 6, 2, 8, 9, 5],
        [1, 0, 0, 5, 0, 0, 0, 0, 7],
        [2, 0, 6, 0, 4, 0, 1, 0, 0],
        [0, 4, 3, 8, 0, 0, 6, 0, 0],
        [0, 8, 0, 7, 2, 6, 9, 0, 0],
    ],
    "sudoku_3.png": [
        [3, 0, 2, 1, 0, 6, 9, 0, 0],
        [0, 8, 0, 0, 0, 0, 0, 0, 2],
        [0, 7, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0],
        [2, 0, 6, 0, 1, 0, 0, 0, 4],
        [0, 0, 0, 4, 0, 0, 7, 0, 0],
        [1, 0, 4, 0, 6, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 9, 0, 5, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0],
    ],
}


def _recognize_sample_grid(image_name: str) -> list[list[int]]:
    image_path = DATA_IMAGE_ROOT / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read sample image: {image_path}")

    grid_image = detect_grid_image(image)
    if grid_image is None:
        raise ValueError(f"Failed to detect Sudoku grid from: {image_path}")

    ocr_grid = prepare_grid_for_ocr(grid_image)
    cells = extract_cells(ocr_grid)
    reader = DigitReader()
    return reader.recognize_grid(cells, threshold=50.0)


@pytest.fixture(scope="module")
def sample_grids() -> dict[str, list[list[int]]]:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip("Tesseract OCR is unavailable in current test environment")

    return {
        image_name: _recognize_sample_grid(image_name)
        for image_name in EXPECTED_SAMPLE_GRIDS
    }


class TestDigitReader:
    """Tests for the digit reader."""

    def test_init(self):
        """Test initializing the digit reader."""
        reader = DigitReader()
        assert reader.config is not None

    def test_recognize_digit_none(self):
        """Test recognizing digit from None image."""
        reader = DigitReader()
        result = reader.recognize_digit(None)
        assert result is None

    def test_recognize_digit_empty_image(self):
        """Test recognizing digit from empty (white) image."""
        empty = np.ones((48, 48), dtype=np.uint8) * 255

        reader = DigitReader()
        result = reader.recognize_digit(empty)

        # Empty image should return None (no digit found)
        assert result is None

    def test_recognize_grid_empty_cells(self):
        """Test recognizing grid with empty cells."""
        # Create 81 empty cells
        empty_cells = [np.ones((48, 48), dtype=np.uint8) * 255] * 81

        reader = DigitReader()
        grid = reader.recognize_grid(empty_cells)

        assert len(grid) == 9
        assert all(len(row) == 9 for row in grid)
        # All cells should be 0 (empty)
        assert all(all(cell == 0 for cell in row) for row in grid)

    def test_recognize_grid_wrong_size(self):
        """Test recognizing grid with wrong number of cells."""
        cells = [np.ones((48, 48), dtype=np.uint8) * 255] * 80

        reader = DigitReader()
        with pytest.raises(ValueError):
            reader.recognize_grid(cells)

    def test_recognize_with_fallback(self):
        """Test recognize with fallback method."""
        empty = np.ones((48, 48), dtype=np.uint8) * 255

        reader = DigitReader()
        result = reader.recognize_with_fallback(empty)

        # Empty image should return None
        assert result is None

    def test_recognize_digit_convenience(self):
        """Test convenience function for recognizing a digit."""
        empty = np.ones((48, 48), dtype=np.uint8) * 255

        result = recognize_digit(empty)

        assert result is None

    def test_recognize_grid_convenience(self):
        """Test convenience function for recognizing a grid."""
        empty_cells = [np.ones((48, 48), dtype=np.uint8) * 255] * 81

        grid = recognize_grid(empty_cells)

        assert len(grid) == 9
        assert all(len(row) == 9 for row in grid)

    def test_preprocess_for_ocr(self):
        """Test preprocessing for OCR."""
        cell = np.ones((48, 48, 3), dtype=np.uint8) * 200

        processed = preprocess_for_ocr(cell)

        assert processed.shape == (64, 64)
        assert processed.dtype == np.uint8

    def test_preprocess_for_ocr_none(self):
        """Test preprocessing None for OCR."""
        processed = preprocess_for_ocr(None)

        assert processed.shape == (64, 64)
        assert np.all(processed == 255)  # Should be all white

    def test_prepare_image(self):
        """Test internal image preparation method."""
        reader = DigitReader()

        # Test with color image
        color = np.ones((48, 48, 3), dtype=np.uint8) * 128
        prepared = reader._prepare_image(color)

        assert prepared.shape == (64, 64)
        assert len(prepared.shape) == 2  # Should be grayscale


class TestDigitReaderRegression:
    """Regression tests for real OCR sample images."""

    def test_recognize_9_in_sudoku_2(self, sample_grids):
        grid = sample_grids["sudoku_2.png"]
        assert grid[4][7] == 9  # r5c8
        assert grid[8][6] == 9  # r9c7

    def test_recognize_9_in_sudoku_3(self, sample_grids):
        grid = sample_grids["sudoku_3.png"]
        assert grid[0][6] == 9  # r1c7
        assert grid[7][5] == 9  # r8c6

    def test_blank_cells_not_false_positive_in_sudoku_3(self, sample_grids):
        grid = sample_grids["sudoku_3.png"]
        assert grid[5][7] == 0  # r6c8
        assert grid[6][7] == 0  # r7c8

    @pytest.mark.parametrize(
        "image_name,minimum_recall",
        [("sudoku_2.png", 37), ("sudoku_3.png", 22)],
    )
    def test_given_digit_recall_floor(self, sample_grids, image_name, minimum_recall):
        predicted = sample_grids[image_name]
        expected = EXPECTED_SAMPLE_GRIDS[image_name]

        matched = 0
        total_givens = 0
        for row in range(9):
            for col in range(9):
                truth = expected[row][col]
                if truth == 0:
                    continue
                total_givens += 1
                if predicted[row][col] == truth:
                    matched += 1

        assert total_givens > 0
        assert matched >= minimum_recall
