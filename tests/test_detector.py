"""Tests for grid detector abstractions."""

import numpy as np

from backend.cv.detector import DetectionResult, warp_grid


class TestDetectionResult:
    def test_fields(self):
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        result = DetectionResult(corners=corners, detector_name="test")
        assert result.detector_name == "test"
        assert result.corners.shape == (4, 2)


class TestWarpGrid:
    def test_warps_to_450(self):
        img = np.ones((500, 500, 3), dtype=np.uint8) * 128
        corners = np.array([[50, 50], [450, 50], [450, 450], [50, 450]], dtype=np.float32)
        warped = warp_grid(img, corners)
        assert warped.shape == (450, 450, 3)

    def test_custom_size(self):
        img = np.ones((500, 500, 3), dtype=np.uint8) * 128
        corners = np.array([[50, 50], [450, 50], [450, 450], [50, 450]], dtype=np.float32)
        warped = warp_grid(img, corners, size=300)
        assert warped.shape == (300, 300, 3)
