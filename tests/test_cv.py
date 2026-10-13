"""Tests for computer vision components."""

import numpy as np
import cv2

from backend.cv.preprocessor import preprocess, morphological_operations
from backend.cv.grid_detector import (
    find_grid_with_corners,
    order_corners,
    perspective_transform,
    has_regular_grid_lines,
)
from backend.cv.cell_extractor import (
    extract_cells,
    clean_cell,
)


class TestPreprocessor:
    """Tests for image preprocessing."""

    def test_preprocess_valid_image(self):
        """Test preprocessing a valid image."""
        # Create a simple test image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = preprocess(img)

        assert result["gray"].shape == (100, 100)
        assert result["thresh"].shape == (100, 100)

    def test_morphological_operations(self):
        """Test morphological operations."""
        thresh = np.ones((100, 100), dtype=np.uint8) * 255

        result = morphological_operations(thresh)

        assert result.shape == thresh.shape
        assert result.dtype == np.uint8


class TestGridDetector:
    """Tests for grid detection."""

    def test_order_corners(self):
        """Test corner ordering."""
        corners = np.array(
            [
                [50, 50],  # Top-left (roughly)
                [150, 40],  # Top-right
                [160, 140],  # Bottom-right
                [30, 150],  # Bottom-left
            ],
            dtype=np.float32,
        )

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
        corners = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)

        result = perspective_transform(img, corners, (50, 50))

        assert result.shape == (50, 50, 3)


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

    def test_clean_cell(self):
        """Test cleaning a cell image."""
        cell = np.ones((48, 48, 3), dtype=np.uint8) * 200

        cleaned = clean_cell(cell)

        assert cleaned.shape == (48, 48)
        assert cleaned.dtype == np.uint8


class TestHasRegularGridLines:
    """Tests for has_regular_grid_lines."""

    def test_synthetic_regular_grid(self):
        """A synthetic 450x450 image with 10 equally-spaced lines should pass."""
        img = np.ones((450, 450), dtype=np.uint8) * 255
        # Draw 10 horizontal and 10 vertical lines at 50px intervals.
        for i in range(10):
            y = i * 50
            cv2.line(img, (0, y), (449, y), 0, 2)
            x = i * 50
            cv2.line(img, (x, 0), (x, 449), 0, 2)
        assert has_regular_grid_lines(img)

    def test_irregular_lines(self):
        """Lines clustered in the centre should fail (insufficient span)."""
        img = np.ones((450, 450), dtype=np.uint8) * 255
        # Draw 10 lines clustered between y=150..285 (span < 70%).
        for i in range(10):
            y = 150 + i * 15
            cv2.line(img, (0, y), (449, y), 0, 2)
            x = 150 + i * 15
            cv2.line(img, (x, 0), (x, 449), 0, 2)
        assert not has_regular_grid_lines(img)

    def test_blank_image(self):
        """A blank image with no lines should fail."""
        img = np.ones((450, 450), dtype=np.uint8) * 255
        assert not has_regular_grid_lines(img)


def test_find_grid_with_corners_with_synthetic_image():
    """Test grid detection with corners does not crash and returns valid shape."""
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (450, 450), (0, 0, 0), 3)

    result = find_grid_with_corners(img)
    if result is None:
        assert result is None
        return

    warped, corners = result
    assert isinstance(warped, np.ndarray)
    assert warped.shape[:2] == (450, 450)
    assert corners.shape == (4, 2)
