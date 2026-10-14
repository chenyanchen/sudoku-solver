from __future__ import annotations

import numpy as np

from backend.cv.corner_cnn_detector import CornerCnnDetector


def test_corners_without_a_grid_are_not_a_detection(monkeypatch):
    detector = CornerCnnDetector.__new__(CornerCnnDetector)
    detector._input_name = "image"

    class Session:
        def run(self, *args):
            return [np.array([[0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9]], dtype=np.float32)]

    monkeypatch.setattr(detector, "_session", Session(), raising=False)
    blank = np.full((450, 450, 3), 255, dtype=np.uint8)

    assert detector.predict(blank) is not None
    assert detector.detect(blank) is None


def test_geometric_confidence_prefers_valid_perspective_quad():
    corners = np.array(
        [[80.0, 120.0], [880.0, 90.0], [930.0, 840.0], [110.0, 890.0]],
        dtype=np.float32,
    )

    score = CornerCnnDetector._geometric_confidence(corners, corners, 1000, 1000)

    assert score > 0.8


def test_geometric_confidence_penalizes_out_of_bounds_predictions():
    corners_raw = np.array(
        [[-120.0, 150.0], [920.0, 80.0], [980.0, 860.0], [60.0, 910.0]],
        dtype=np.float32,
    )
    corners_clipped = np.array(
        [[0.0, 150.0], [920.0, 80.0], [980.0, 860.0], [60.0, 910.0]],
        dtype=np.float32,
    )

    score = CornerCnnDetector._geometric_confidence(corners_raw, corners_clipped, 1000, 1000)

    assert score < 0.7

