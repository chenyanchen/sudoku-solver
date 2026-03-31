"""Tests for ML backend release-model configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import ml_backend.app as ml_backend_app
from backend.cv.detector import DetectionResult


def _write_meta(path: Path, version: str) -> None:
    path.write_text(json.dumps({"model_version": version}), encoding="utf-8")


def test_validate_backend_model_env_requires_explicit_paths(monkeypatch):
    monkeypatch.delenv("CNN_MODEL_PATH", raising=False)
    monkeypatch.delenv("YOLO_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="CNN_MODEL_PATH"):
        ml_backend_app._validate_backend_model_env()


def test_setup_returns_release_model_version_from_metadata(tmp_path, monkeypatch):
    cnn_path = tmp_path / "cnn.onnx"
    yolo_path = tmp_path / "yolo.onnx"
    cnn_path.write_bytes(b"cnn")
    yolo_path.write_bytes(b"yolo")
    _write_meta(cnn_path.with_suffix(".meta.json"), "20260331.1")
    _write_meta(yolo_path.with_suffix(".meta.json"), "20260331.2")

    monkeypatch.setenv("CNN_MODEL_PATH", str(cnn_path))
    monkeypatch.setenv("YOLO_MODEL", str(yolo_path))
    monkeypatch.setattr(ml_backend_app, "_get_cnn", lambda: type("C", (), {"is_ready": True})())
    monkeypatch.setattr(ml_backend_app, "_get_yolo", lambda: type("Y", (), {"is_ready": True})())

    payload = {"label_config": "<View><Image name='image' value='$image'/></View>"}
    assert ml_backend_app.setup(payload) == {"model_version": "20260331.1+20260331.2"}


def test_predict_uses_release_model_version_in_results(tmp_path, monkeypatch):
    cnn_path = tmp_path / "cnn.onnx"
    yolo_path = tmp_path / "yolo.onnx"
    cnn_path.write_bytes(b"cnn")
    yolo_path.write_bytes(b"yolo")
    _write_meta(cnn_path.with_suffix(".meta.json"), "20260331.1")
    _write_meta(yolo_path.with_suffix(".meta.json"), "20260331.2")

    monkeypatch.setenv("CNN_MODEL_PATH", str(cnn_path))
    monkeypatch.setenv("YOLO_MODEL", str(yolo_path))
    monkeypatch.setattr(ml_backend_app, "_decode_image", lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(ml_backend_app, "_predict_ocr", lambda image: ([{"x": 1}], 0.75))

    response = ml_backend_app.predict(
        {
            "label_config": "<View><Image name='image' value='$image'/></View>",
            "tasks": [{"data": {"image": "/data/upload/test.jpg"}}],
        }
    )

    assert response == {
        "results": [
            {
                "result": [{"x": 1}],
                "score": 0.75,
                "model_version": "20260331.1+20260331.2",
            }
        ]
    }


def test_predict_keypoints_returns_model_16_keypoints_directly(tmp_path, monkeypatch):
    yolo_path = tmp_path / "yolo.onnx"
    cnn_path = tmp_path / "cnn.onnx"
    yolo_path.write_bytes(b"yolo")
    cnn_path.write_bytes(b"cnn")
    _write_meta(cnn_path.with_suffix(".meta.json"), "20260331.1")
    _write_meta(yolo_path.with_suffix(".meta.json"), "20260331.2")

    monkeypatch.setenv("CNN_MODEL_PATH", str(cnn_path))
    monkeypatch.setenv("YOLO_MODEL", str(yolo_path))
    monkeypatch.setattr(
        ml_backend_app,
        "_decode_image",
        lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8),
    )

    keypoints = [[10.0 + i, 20.0 + i] for i in range(16)]

    class FakeYolo:
        is_ready = True
        last_confidence = 0.88

        def detect(self, image):
            return DetectionResult(
                corners=np.array(keypoints[:4], dtype=np.float32),
                detector_name="yolo",
                keypoints=np.array(keypoints, dtype=np.float32),
            )

    monkeypatch.setattr(ml_backend_app, "_get_yolo", lambda: FakeYolo())

    response = ml_backend_app.predict(
        {
            "label_config": "<View><Image name='image' value='$image'/><KeyPointLabels name='corners' toName='image'></KeyPointLabels></View>",
            "tasks": [{"data": {"image": "/data/upload/test.jpg"}}],
        }
    )

    result = response["results"][0]
    assert result["score"] == 0.88
    assert result["model_version"] == "20260331.2"
    assert len(result["result"]) == 16
    assert result["result"][0]["value"]["x"] == 10.0
    assert result["result"][0]["value"]["y"] == 20.0
    assert result["result"][15]["value"]["x"] == 25.0
    assert result["result"][15]["value"]["y"] == 35.0
