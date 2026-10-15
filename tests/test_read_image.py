"""Web reading order: OpenCV first, Corner CNN when OpenCV's grid does not solve."""

import numpy as np
import pytest

from backend.api import routes
from backend.cv.detector import DetectionResult
from backend.ocr.pipeline import AMBIGUOUS, UNIQUE, Recognition

QUAD = np.float32([[10, 10], [90, 10], [90, 90], [10, 90]])
IMAGE = np.full((100, 100, 3), 255, np.uint8)


class OpenCV:
    def __init__(self, found):
        self.found = found

    def detect(self, image):
        return DetectionResult(QUAD, "opencv") if self.found else None


class CornerCnn:
    calls = 0

    def predict(self, image):
        CornerCnn.calls += 1
        return QUAD


@pytest.fixture
def setup(monkeypatch):
    def configure(opencv_found, statuses, regular=True):
        CornerCnn.calls = 0
        outcomes = iter(statuses)
        monkeypatch.setattr(routes, "_get_cnn_reader", lambda: None)
        monkeypatch.setattr(routes, "_OPENCV", OpenCV(opencv_found))
        monkeypatch.setattr(routes, "_get_corner_cnn", lambda: CornerCnn())
        monkeypatch.setattr(routes, "recognize", lambda reader, warped: Recognition([], None, next(outcomes), {}))
        monkeypatch.setattr(routes, "has_regular_grid_lines", lambda gray: regular)
    return configure


def test_solved_by_opencv_skips_corner_cnn(setup):
    setup(True, [UNIQUE])
    assert routes.read_image(IMAGE).detector_name == "opencv"
    assert CornerCnn.calls == 0


def test_unsolved_opencv_grid_is_retried_with_corner_cnn(setup):
    setup(True, [AMBIGUOUS, UNIQUE])
    assert routes.read_image(IMAGE).detector_name == "corner_cnn"


def test_neither_solves_keeps_opencv_reading(setup):
    setup(True, [AMBIGUOUS, AMBIGUOUS])
    assert routes.read_image(IMAGE).detector_name == "opencv"


def test_unsolved_corner_cnn_quad_counts_only_if_regular(setup):
    setup(False, [AMBIGUOUS], regular=False)
    assert routes.read_image(IMAGE) is None
    setup(False, [AMBIGUOUS], regular=True)
    assert routes.read_image(IMAGE).detector_name == "corner_cnn"
