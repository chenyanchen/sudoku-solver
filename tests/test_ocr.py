"""Tests for OCR components."""

import pytest
import numpy as np
import cv2

from backend.ocr.digit_reader import (
    DigitReader,
    recognize_digit,
    recognize_grid,
    preprocess_for_ocr,
)


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
