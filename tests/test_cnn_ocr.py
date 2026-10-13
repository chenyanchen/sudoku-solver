"""Tests for CNN OCR reader behavior."""

from pathlib import Path

import numpy as np
import pytest

from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.solver.backtracking import is_valid_placement


def test_cnn_reader_raises_for_missing_model(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CnnDigitReader(model_path=tmp_path / "missing.onnx")


def test_uncertain_digit_is_not_promoted_by_recheck(monkeypatch):
    reader = CnnDigitReader.__new__(CnnDigitReader)
    reader._input_size = 48
    reader._class_values = list(range(10))
    reader.blank_threshold, reader.digit_threshold, reader.top_k_candidates = 0.65, 0.55, 4
    reader._model_version = "test"
    probs = np.array([0.30, 0.40] + [0.30 / 8] * 8, dtype=np.float32)
    monkeypatch.setattr(reader, "_predict_probs_batch", lambda images: np.tile(probs, (len(images), 1)))
    cell = np.full((48, 48), 255, dtype=np.uint8)
    assert reader.recognize_grid_with_metadata([cell] * 81)[0] == [[0] * 9 for _ in range(9)]
    # The call-specific threshold must still be honored.
    assert reader.recognize_grid_with_metadata([cell] * 81, threshold=0.35)[0][0][0] == 1


def test_is_valid_placement_checks_row_col_box():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 7
    grid[1][1] = 7

    assert is_valid_placement(grid, 0, 2, 7) is False  # row conflict
    assert is_valid_placement(grid, 2, 0, 7) is False  # col conflict
    assert is_valid_placement(grid, 2, 2, 7) is False  # box conflict
    assert is_valid_placement(grid, 4, 4, 7) is True
