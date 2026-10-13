"""Tests for ML backend release-model configuration."""

from __future__ import annotations

import json

import pytest

from ml_backend import config


def _release(path, version):
    path.write_bytes(b"model")
    path.with_suffix(".meta.json").write_text(json.dumps({"model_version": version}))
    return path


def test_validate_backend_model_env_requires_explicit_paths(monkeypatch):
    monkeypatch.delenv("CNN_MODEL_PATH", raising=False)
    monkeypatch.delenv("CORNER_CNN_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="CNN_MODEL_PATH, CORNER_CNN_MODEL"):
        config.validate_backend_model_env()


def test_validate_backend_model_env_requires_metadata_sidecar(tmp_path, monkeypatch):
    cnn = _release(tmp_path / "cnn.onnx", "20260331.1")
    corner = tmp_path / "corner.onnx"
    corner.write_bytes(b"model")
    monkeypatch.setenv("CNN_MODEL_PATH", str(cnn))
    monkeypatch.setenv("CORNER_CNN_MODEL", str(corner))

    with pytest.raises(RuntimeError, match="Release metadata file is missing"):
        config.validate_backend_model_env()


def test_current_release_versions_reads_sidecars(tmp_path, monkeypatch):
    monkeypatch.setenv("CNN_MODEL_PATH", str(_release(tmp_path / "cnn.onnx", "20260331.1")))
    monkeypatch.setenv("CORNER_CNN_MODEL", str(_release(tmp_path / "corner.onnx", "20260331.2")))

    assert config.current_release_versions() == ("20260331.1", "20260331.2")
