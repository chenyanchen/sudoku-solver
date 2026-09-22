from __future__ import annotations

import pytest
import numpy as np

from backend.cv.corner_cnn_detector import CornerCnnDetector


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


def test_geometric_confidence_returns_zero_for_non_convex_quad():
    corners = np.array(
        [[100.0, 100.0], [900.0, 100.0], [150.0, 850.0], [850.0, 850.0]],
        dtype=np.float32,
    )

    score = CornerCnnDetector._geometric_confidence(corners, corners, 1000, 1000)

    assert score == pytest.approx(0.0)
