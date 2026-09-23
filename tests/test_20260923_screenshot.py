"""Regression for the Sep 23 browser screenshot (local fixture supplied by user)."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.solver.backtracking import SudokuSolver

IMAGE = Path(__file__).resolve().parents[1] / "data/raw/images/Screenshot_20260923111925.png"
EXPECTED = [
    [0, 9, 0, 0, 4, 3, 0, 0, 8],
    [5, 0, 7, 0, 0, 9, 6, 0, 0],
    [0, 0, 0, 2, 0, 0, 3, 1, 9],
    [0, 0, 1, 0, 9, 0, 0, 4, 2],
    [0, 0, 8, 4, 0, 6, 1, 0, 0],
    [3, 5, 0, 0, 2, 0, 9, 0, 0],
    [2, 8, 5, 0, 0, 7, 0, 0, 0],
    [0, 0, 3, 5, 0, 0, 2, 0, 7],
    [4, 0, 0, 1, 6, 0, 0, 3, 0],
]


def test_screenshot_recognizes_and_solves():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/sudoku:recognize",
            files={"image": (IMAGE.name, IMAGE.read_bytes(), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["success"], response.json()
        grid = response.json()["grid"]
        assert grid == EXPECTED
        solved = client.post("/api/v1/sudoku:solve", json={"grid": {"cells": grid}}).json()
        assert solved["success"]
        assert solved["solved"] == SudokuSolver().solve(EXPECTED)
        assert SudokuSolver().count_solutions(grid, max_count=2) == 1
