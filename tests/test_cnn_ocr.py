"""Tests for CNN OCR reader behavior."""

from pathlib import Path

import numpy as np
import pytest

from backend.api import routes
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.solver.backtracking import is_valid_placement


def test_cnn_reader_not_ready_for_missing_model(tmp_path: Path):
    model_path = tmp_path / "missing.onnx"
    reader = CnnDigitReader(model_path=model_path, strict=False)

    assert reader.is_ready is False
    assert reader.load_error is not None


def test_cnn_reader_raises_when_not_ready(tmp_path: Path):
    model_path = tmp_path / "missing.onnx"
    reader = CnnDigitReader(model_path=model_path, strict=False)

    with pytest.raises(RuntimeError):
        reader.recognize_grid([None] * 81)


def test_uncertain_digit_is_not_promoted_by_recheck(monkeypatch):
    reader = CnnDigitReader(model_path="missing.onnx")
    reader._class_values = list(range(10))
    probs = np.array([0.30, 0.40] + [0.30 / 8] * 8, dtype=np.float32)
    monkeypatch.setattr(reader, "_ensure_ready", lambda: None)
    monkeypatch.setattr(reader, "_predict_probs", lambda image: probs)
    monkeypatch.setattr(reader, "_predict_probs_batch", lambda images: np.tile(probs, (len(images), 1)))
    cell = np.full((48, 48), 255, dtype=np.uint8)
    assert reader.predict_cell(cell)[0] is None
    assert reader.recognize_grid([cell] * 81) == [[0] * 9 for _ in range(9)]
    # The call-specific threshold must still be honored.
    assert reader.recognize_grid([cell] * 81, threshold=0.35)[0][0] == 1


def test_is_valid_placement_checks_row_col_box():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 7
    grid[1][1] = 7

    assert is_valid_placement(grid, 0, 2, 7) is False  # row conflict
    assert is_valid_placement(grid, 2, 0, 7) is False  # col conflict
    assert is_valid_placement(grid, 2, 2, 7) is False  # box conflict
    assert is_valid_placement(grid, 4, 4, 7) is True


def test_resolve_ocr_reader_fallback_to_cnn(monkeypatch):
    dummy_reader = object()
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    monkeypatch.setattr(routes, "_get_cnn_reader", lambda: (dummy_reader, None))

    reader, engine, err = routes._resolve_ocr_reader()

    assert reader is dummy_reader
    assert engine == "cnn"
    assert err is None
