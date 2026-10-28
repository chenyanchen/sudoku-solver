"""Web reading order: OpenCV first, then Corner CNN, then Corner CNN refined on a crop."""

import numpy as np
import pytest

from backend.api import routes
from backend.cv.detector import DetectionResult
from backend.ocr.pipeline import AMBIGUOUS, UNIQUE, UNSOLVABLE, Recognition

QUAD = np.float32([[10, 10], [90, 10], [90, 90], [10, 90]])
OPENCV_QUAD = np.float32([[20, 20], [40, 20], [40, 40], [20, 40]])
IMAGE = np.full((100, 100, 3), 255, np.uint8)


class OpenCV:
    def __init__(self, found):
        self.found = found

    def detect(self, image):
        return DetectionResult(OPENCV_QUAD, "opencv") if self.found else None


class CornerCnn:
    calls = 0
    refines = 0

    def predict(self, image):
        CornerCnn.calls += 1
        return QUAD

    def refine(self, image, corners):
        CornerCnn.refines += 1
        return QUAD


def _recognition(outcome):
    """A status, (status, number of givens), or (status, givens, solution)."""
    status, givens, solution = (outcome + (None,))[:3] if isinstance(outcome, tuple) else (outcome, 0, None)
    grid = [[1 if r * 9 + c < givens else 0 for c in range(9)] for r in range(9)]
    return Recognition(grid, solution, status, {})


@pytest.fixture
def setup(monkeypatch):
    def configure(opencv_found, statuses, regular=True):
        """`regular`: whether every warp has regular grid lines, or the detectors whose do."""
        CornerCnn.calls = CornerCnn.refines = 0
        outcomes = iter(statuses)
        monkeypatch.setattr(routes, "_get_cnn_reader", lambda: None)
        monkeypatch.setattr(routes, "_OPENCV", OpenCV(opencv_found))
        monkeypatch.setattr(routes, "_get_corner_cnn", lambda: CornerCnn())
        monkeypatch.setattr(routes, "recognize", lambda reader, warped: _recognition(next(outcomes)))
        # Each warp is filled with the value of the detector whose quad it came from.
        monkeypatch.setattr(
            routes, "warp_grid",
            lambda image, corners: np.full((450, 450, 3), 1 if corners is OPENCV_QUAD else 2, np.uint8),
        )
        names = {1: "opencv", 2: "corner_cnn"}
        monkeypatch.setattr(
            routes, "has_regular_grid_lines",
            lambda gray: regular if isinstance(regular, bool) else names[gray[0, 0]] in regular,
        )
    return configure


def test_solved_by_opencv_skips_corner_cnn(setup):
    setup(True, [UNIQUE])
    assert routes.read_image(IMAGE).detector_name == "opencv"
    assert CornerCnn.calls == 0


def test_unsolved_opencv_grid_is_retried_with_corner_cnn(setup):
    setup(True, [AMBIGUOUS, UNIQUE, AMBIGUOUS])
    assert routes.read_image(IMAGE).detector_name == "corner_cnn"


SOLUTION = [[1] * 9] * 9
OTHER_SOLUTION = [[2] * 9] * 9


def test_refined_reading_with_more_givens_of_same_solution_wins(setup):
    """A quad a little off cuts edge digits; the givens left may still solve."""
    setup(False, [(UNIQUE, 30, SOLUTION), (UNIQUE, 33, SOLUTION)])
    assert routes.read_image(IMAGE).recognition.givens == 33


@pytest.mark.parametrize("refined", [
    (UNIQUE, 33, OTHER_SOLUTION),  # one reading is wrong; trust the first quad
    (UNIQUE, 30, SOLUTION),
    (UNSOLVABLE, 33),
])
def test_first_corner_cnn_reading_that_solves_is_kept(setup, refined):
    setup(False, [(UNIQUE, 30, SOLUTION), refined])
    reading = routes.read_image(IMAGE)
    assert reading.recognition.status == UNIQUE
    assert reading.recognition.givens == 30
    assert reading.recognition.solution == SOLUTION


def test_unsolved_corner_cnn_quad_is_retried_refined(setup):
    setup(True, [AMBIGUOUS, AMBIGUOUS, UNIQUE])
    assert routes.read_image(IMAGE).recognition.status == UNIQUE
    assert CornerCnn.refines == 1


def test_neither_solves_keeps_opencv_reading(setup):
    setup(True, [AMBIGUOUS, AMBIGUOUS, AMBIGUOUS])
    assert routes.read_image(IMAGE).detector_name == "opencv"


def test_unsolved_corner_cnn_quad_counts_only_if_regular(setup):
    setup(False, [AMBIGUOUS, AMBIGUOUS], regular=False)
    assert routes.read_image(IMAGE) is None
    setup(False, [AMBIGUOUS, AMBIGUOUS], regular=True)
    assert routes.read_image(IMAGE).detector_name == "corner_cnn"


def test_ambiguous_reading_beats_unsolvable_one(setup):
    """An ambiguous puzzle never reads as unique; an unsolvable reading is certainly wrong."""
    setup(True, [UNSOLVABLE, (AMBIGUOUS, 34), (AMBIGUOUS, 34)])
    assert routes.read_image(IMAGE).detector_name == "corner_cnn"


def test_partial_grid_does_not_beat_unsolvable_reading(setup):
    """A quad over part of the grid reads few givens, which have solutions too."""
    setup(True, [UNSOLVABLE, (AMBIGUOUS, 5), (AMBIGUOUS, 16)])
    assert routes.read_image(IMAGE).detector_name == "opencv"


def test_regular_corner_cnn_grid_beats_irregular_opencv_quad(setup):
    """OpenCV's fallback quad may be one cell: its blank reading is no use."""
    setup(True, [AMBIGUOUS, UNSOLVABLE, UNSOLVABLE], regular={"corner_cnn"})
    assert routes.read_image(IMAGE).detector_name == "corner_cnn"
