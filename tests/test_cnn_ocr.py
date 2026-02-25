"""Tests for CNN OCR reader behavior."""

from pathlib import Path

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


def test_constraint_rerank_resolves_low_conflict():
    reader = CnnDigitReader.__new__(CnnDigitReader)
    reader.rerank_confidence = 0.8

    # Start from an invalid row duplication: [5, 5, 0, ...]
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    grid[0][1] = 5

    confidences = [0.95] * 81
    confidences[1] = 0.25  # low confidence duplicated cell

    candidates = [[] for _ in range(81)]
    candidates[1] = [(3, 0.96), (5, 0.90)]

    reranked = reader._rerank_with_constraints(grid, confidences, candidates)

    assert reranked[0][0] == 5
    assert reranked[0][1] == 3


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
