"""Tests for keypoint fit validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import ml_backend.app as ml_backend_app


def _annotation_with_duplicate_label() -> dict:
    points = []
    labels = [f"P{r}{c}" for r in range(4) for c in range(4)]
    labels[6] = "P13"  # duplicate label, P12 missing
    idx = 0
    for row in range(4):
        for col in range(4):
            points.append(
                {
                    "type": "keypointlabels",
                    "value": {
                        "x": 10 + col * 20,
                        "y": 10 + row * 20,
                        "keypointlabels": [labels[idx]],
                    },
                }
            )
            idx += 1
    return {"result": points}


def test_fit_keypoints_rejects_duplicate_labels_and_logs_missing_keypoint(
    tmp_path, monkeypatch, caplog
):
    repo_root = tmp_path
    image_dir = repo_root / "data" / "raw" / "images"
    label_dir = repo_root / "data" / "yolo_grid" / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "IMG_8711.jpg"), image)

    monkeypatch.setattr(ml_backend_app, "REPO_ROOT", repo_root)
    task = {"data": {"image": "/data/upload/2/d351c48c-IMG_8711.jpg"}}

    ml_backend_app._fit_keypoints(_annotation_with_duplicate_label(), task)

    assert not (label_dir / "IMG_8711.txt").exists()
    assert "missing=['P12']" in caplog.text
    assert "duplicates=['P13']" in caplog.text
