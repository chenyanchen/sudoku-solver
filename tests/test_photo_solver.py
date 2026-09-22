"""Known-photo regression, not an independent generalization benchmark."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app
from backend.solver.backtracking import SudokuSolver

ROOT = Path(__file__).resolve().parents[1]


def test_ordinary_photos_recognize_and_solve(monkeypatch):
    labels = json.loads((ROOT / "data/labels/sudoku_labels.json").read_text())
    monkeypatch.setattr(routes, "_CNN_READER", None)
    monkeypatch.setattr(routes, "_DETECTOR", None)
    with TestClient(app) as client:
        # 8695–8708: ordinary photographed screens. 8709–8712 have vine borders.
        for number in range(8695, 8709):
            name = f"IMG_{number}.jpg"
            response = client.post(
                "/api/v1/sudoku:recognize",
                files={"image": (name, (ROOT / "data/raw/images" / name).read_bytes(), "image/jpeg")},
            )
            assert response.status_code == 200, name
            recognized = response.json()
            assert recognized["success"], (name, recognized["message"])
            assert recognized["grid"] == labels[name], name
            solved = client.post("/api/v1/sudoku:solve", json={"grid": {"cells": recognized["grid"]}}).json()
            assert solved["success"], name
            assert solved["solved"] == SudokuSolver().solve(labels[name]), name
