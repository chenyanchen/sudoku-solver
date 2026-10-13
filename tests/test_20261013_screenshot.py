"""Regression for the Oct 13 screenshot: shaded-cell borders hid 14 givens."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.solver.backtracking import SudokuSolver

IMAGE = Path(__file__).resolve().parents[1] / "data/raw/images/Screenshot_20261013213520.png"
EXPECTED = [
    [0, 5, 0, 8, 7, 1, 9, 3, 0],
    [9, 7, 0, 0, 2, 6, 0, 0, 0],
    [0, 8, 2, 0, 0, 9, 6, 7, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 5],
    [0, 1, 0, 7, 0, 0, 4, 6, 9],
    [4, 0, 7, 9, 1, 5, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0],
    [3, 9, 1, 0, 4, 0, 0, 5, 0],
    [5, 0, 6, 1, 0, 0, 0, 0, 7],
]


def test_screenshot_recognizes_all_givens():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sudoku:recognize",
            files={"image": (IMAGE.name, IMAGE.read_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["success"], response.json()
        assert response.json()["grid"] == EXPECTED
        assert SudokuSolver().count_solutions(EXPECTED, max_count=2) == 1
