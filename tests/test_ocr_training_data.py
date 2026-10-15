"""Training must not inherit detector mistakes or split cells of one puzzle."""

import json

import cv2
import numpy as np
import pytest

from backend.solver.backtracking import SudokuSolver, is_valid_grid
from scripts._paths import DEFAULT_CORNERS, load_corner_labels
from scripts.train_cnn_ocr import REPO_ROOT, CellSample, build_samples, grouped_split

LABELS = json.loads((REPO_ROOT / "data" / "labels" / "sudoku_labels.json").read_text())


def test_training_uses_labeled_corners_and_keeps_puzzles_together(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(images / "board.jpg"), image)
    labels = tmp_path / "ocr.json"
    labels.write_text(json.dumps({"images/board.jpg": [[0] * 9 for _ in range(9)]}))
    corners = tmp_path / "labels.csv"
    corners.write_text(
        "image,x_tl,y_tl,x_tr,y_tr,x_br,y_br,x_bl,y_bl\n"
        "board.jpg,0,0,0.99,0,0.99,0.99,0,0.99\n"
    )
    samples = build_samples(labels, 48, corners_path=corners)
    assert len(samples) == 81
    assert all(s.label == 0 and np.all(s.image == 255) for s in samples)
    samples += [CellSample(np.zeros((48, 48), dtype=np.uint8), 1, "other")] * 81
    train, val = grouped_split(samples, 0.2, 42)
    assert len(train) == len(val) == 81
    assert {s.source for s in train}.isdisjoint(s.source for s in val)
    assert [s.source for s in val] == [s.source for s in grouped_split(samples, 0.2, 42)[1]]

    corners.write_text("image,x_tl,y_tl,x_tr,y_tr,x_br,y_br,x_bl,y_bl\n")
    with pytest.raises(ValueError, match="Missing ground-truth corners"):
        build_samples(labels, 48, corners_path=corners)


def test_labels_are_solvable_puzzles_with_reviewed_corners():
    corners = load_corner_labels(DEFAULT_CORNERS)
    for name, grid in LABELS.items():
        assert is_valid_grid(grid), name
        # 0 solutions means a transcription error; >1 is allowed (an ambiguous puzzle).
        assert SudokuSolver().count_solutions(grid) >= 1, name
        assert name.rsplit("/", 1)[-1] in corners, name
