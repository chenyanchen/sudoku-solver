"""Startup fails fast when the CNN model cannot be loaded."""

import pytest
from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app


def test_startup_fails_on_missing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "_CNN_READER", None)
    monkeypatch.setenv("CNN_MODEL_PATH", str(tmp_path / "missing.onnx"))

    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass


def test_startup_loads_configured_model(monkeypatch):
    monkeypatch.setattr(routes, "_CNN_READER", None)

    with TestClient(app) as client:
        assert client.get("/health").json()["cnn_model_loaded"] is True
