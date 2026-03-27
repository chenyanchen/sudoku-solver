"""Tests for grid detector abstractions."""

import numpy as np

from backend.cv.detector import DetectionResult, ChainDetector, warp_grid


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


class TestChainDetector:
    def test_returns_first_success(self):
        class AlwaysNone:
            def detect(self, image):
                return None

        class AlwaysFound:
            def detect(self, image):
                corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
                return DetectionResult(corners=corners, detector_name="found")

        chain = ChainDetector([AlwaysNone(), AlwaysFound()])
        result = chain.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is not None
        assert result.detector_name == "found"

    def test_returns_none_when_all_fail(self):
        class AlwaysNone:
            def detect(self, image):
                return None

        chain = ChainDetector([AlwaysNone(), AlwaysNone()])
        assert chain.detect(np.zeros((100, 100, 3), dtype=np.uint8)) is None

    def test_stops_at_first_success(self):
        call_count = [0, 0]

        class Detector0:
            def detect(self, image):
                call_count[0] += 1
                corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
                return DetectionResult(corners=corners, detector_name="d0")

        class Detector1:
            def detect(self, image):
                call_count[1] += 1
                return None

        chain = ChainDetector([Detector0(), Detector1()])
        chain.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert call_count == [1, 0]
