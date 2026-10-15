"""The strict OCR eval is the release gate: it passes only when every label was checked."""

import json
import os
import sys

from scripts import eval_cnn_ocr


def _strict(monkeypatch, tmp_path, labels):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(labels))
    monkeypatch.setattr(sys, "argv", [
        "eval_cnn_ocr.py", "--cnn-model", os.environ["CNN_MODEL_PATH"], "--labels", str(path), "--strict",
    ])
    return eval_cnn_ocr.main()


def test_strict_eval_fails_when_a_labeled_image_is_missing(monkeypatch, tmp_path):
    grid = json.loads((eval_cnn_ocr.REPO_ROOT / "data/labels/sudoku_labels.json").read_text())["IMG_8695.jpg"]
    assert _strict(monkeypatch, tmp_path, {"IMG_8695.jpg": grid}) == 0
    assert _strict(monkeypatch, tmp_path, {"IMG_8695.jpg": grid, "missing.jpg": grid}) == 1
