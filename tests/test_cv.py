"""Tests for computer vision components."""

import pytest
import numpy as np
import cv2

from backend.cv.preprocessor import preprocess, enhance_contrast, denoise, morphological_operations, invert_image
from backend.cv.grid_detector import find_grid, order_corners, perspective_transform, validate_grid
from backend.cv.cell_extractor import extract_cells, clean_cell, extract_digit, get_cell_grid_positions


class TestPreprocessor:
    """Tests for image preprocessing."""

    def test_preprocess_valid_image(self):
        """Test preprocessing a valid image."""
        # Create a simple test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = preprocess(img)

        assert 'gray' in result
        assert 'blurred' in result
        assert 'thresh' in result
        assert 'original' in result

        assert result['gray'].shape == (100, 100)
        assert result['blurred'].shape == (100, 100)
        assert result['thresh'].shape == (100, 100)

    def test_preprocess_none_image(self):
        """Test preprocessing with None image raises error."""
        with pytest.raises(ValueError):
            preprocess(None)

    def test_enhance_contrast(self):
        """Test contrast enhancement."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128

        result = enhance_contrast(gray)

        assert result.shape == gray.shape
        assert result.dtype == np.uint8

    def test_denoise(self):
        """Test denoising."""
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        result = denoise(gray)

        assert result.shape == gray.shape
        assert result.dtype == np.uint8

    def test_morphological_operations(self):
        """Test morphological operations."""
        thresh = np.ones((100, 100), dtype=np.uint8) * 255

        result = morphological_operations(thresh)

        assert result.shape == thresh.shape
        assert result.dtype == np.uint8

    def test_invert_image(self):
        """Test image inversion."""
        img = np.ones((100, 100), dtype=np.uint8) * 200

        result = invert_image(img)

        assert result.shape == img.shape
        # 255 - 200 = 55
        assert result[0, 0] == 55


class TestGridDetector:
    """Tests for grid detection."""

    def test_order_corners(self):
        """Test corner ordering."""
        corners = np.array([
            [50, 50],   # Top-left (roughly)
            [150, 40],  # Top-right
            [160, 140], # Bottom-right
            [30, 150]   # Bottom-left
        ], dtype=np.float32)

        ordered = order_corners(corners)

        assert ordered.shape == (4, 2)
        assert ordered.dtype == np.float32

        # Top-left should have smallest x and y
        top_left = ordered[0]
        assert top_left[0] < ordered[1][0]  # Less than top-right x
        assert top_left[1] < ordered[3][1]  # Less than bottom-left y

    def test_perspective_transform(self):
        """Test perspective transformation."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        corners = np.array([
            [10, 10],
            [90, 10],
            [90, 90],
            [10, 90]
        ], dtype=np.float32)

        result = perspective_transform(img, corners, (50, 50))

        assert result.shape == (50, 50, 3)

    def test_validate_grid_none(self):
        """Test grid validation with None."""
        assert validate_grid(None) is False

    def test_validate_grid_too_small(self):
        """Test grid validation with small image."""
        small = np.ones((50, 50, 3), dtype=np.uint8)
        assert validate_grid(small) is False


class TestCellExtractor:
    """Tests for cell extraction."""

    def test_extract_cells_from_grid(self):
        """Test extracting cells from a grid image."""
        # Create a 450x450 grid image
        grid = np.ones((450, 450, 3), dtype=np.uint8) * 255

        cells = extract_cells(grid)

        assert len(cells) == 81
        # Each cell should be 48x48 (50 - 2 for borders)
        assert all(cell.shape == (48, 48, 3) for cell in cells if cell.size > 0)

    def test_extract_cells_from_none(self):
        """Test extracting cells from None image."""
        cells = extract_cells(None)
        assert cells == []

    def test_clean_cell(self):
        """Test cleaning a cell image."""
        cell = np.ones((48, 48, 3), dtype=np.uint8) * 200

        cleaned = clean_cell(cell)

        assert cleaned.shape == (48, 48)
        assert cleaned.dtype == np.uint8

    def test_clean_cell_none(self):
        """Test cleaning None cell returns blank image."""
        cleaned = clean_cell(None)

        assert cleaned.shape == (48, 48)
        assert np.all(cleaned == 255)  # Should be all white

    def test_extract_digit_from_blank(self):
        """Test extracting digit from blank cell."""
        blank = np.ones((48, 48, 3), dtype=np.uint8) * 255

        digit, has_digit = extract_digit(blank)

        assert has_digit is False
        assert digit.shape == (48, 48)

    def test_extract_cell_grid_positions(self):
        """Test getting grid positions."""
        positions = get_cell_grid_positions(9)

        assert len(positions) == 81
        # First position should be at origin
        assert positions[0] == (0, 0, 50, 50)


def test_find_grid_with_synthetic_image():
    """Test grid detection with a synthetic grid image."""
    # Create a simple grid-like image
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255

    # Draw a rectangle
    cv2.rectangle(img, (50, 50), (450, 450), (0, 0, 0), 3)

    # This won't detect as a proper Sudoku grid but tests the function doesn't crash
    result = find_grid(img)

    # Result may be None (no valid grid) or an image
    assert result is None or isinstance(result, np.ndarray)
