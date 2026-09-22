from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts import export_corner_cnn_onnx


def test_parse_args_defaults_to_opset_18(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_corner_cnn_onnx.py",
            "--checkpoint",
            "checkpoint.pt",
            "--output",
            "model.onnx",
        ],
    )

    args = export_corner_cnn_onnx.parse_args()

    assert args.opset == 18


def test_export_model_uses_dynamic_shapes(monkeypatch, tmp_path: Path):
    payload = {"state_dict": {}, "input_size": 256}
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "model.onnx"
    checkpoint.write_bytes(b"stub")

    class DummyModel:
        def load_state_dict(self, state_dict):
            assert state_dict == {}

        def eval(self):
            return self

    captured: dict[str, object] = {}

    def fake_export(model, args, f, **kwargs):
        captured["kwargs"] = kwargs
        output.write_bytes(b"fake-onnx")

    monkeypatch.setattr(export_corner_cnn_onnx.torch, "load", lambda *a, **k: payload)
    monkeypatch.setattr(export_corner_cnn_onnx, "create_corner_model", lambda: DummyModel())
    monkeypatch.setattr(export_corner_cnn_onnx.torch.onnx, "export", fake_export)
    monkeypatch.setattr(
        export_corner_cnn_onnx,
        "_load_onnx_opset",
        lambda path: 18,
    )

    args = argparse.Namespace(checkpoint=checkpoint, output=output, opset=18)

    assert export_corner_cnn_onnx.export_checkpoint(args) == output

    kwargs = captured["kwargs"]
    assert kwargs["opset_version"] == 18
    assert kwargs["dynamic_axes"] is None
    assert "dynamic_shapes" in kwargs
    assert isinstance(kwargs["dynamic_shapes"], tuple)
    assert kwargs["dynamic_shapes"][0][0] is not None


def test_export_checkpoint_rejects_opset_mismatch(monkeypatch, tmp_path: Path):
    payload = {"state_dict": {}, "input_size": 256}
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "model.onnx"
    checkpoint.write_bytes(b"stub")

    class DummyModel:
        def load_state_dict(self, state_dict):
            assert state_dict == {}

        def eval(self):
            return self

    monkeypatch.setattr(export_corner_cnn_onnx.torch, "load", lambda *a, **k: payload)
    monkeypatch.setattr(export_corner_cnn_onnx, "create_corner_model", lambda: DummyModel())
    monkeypatch.setattr(
        export_corner_cnn_onnx.torch.onnx,
        "export",
        lambda *a, **k: output.write_bytes(b"fake-onnx"),
    )
    monkeypatch.setattr(
        export_corner_cnn_onnx,
        "_load_onnx_opset",
        lambda path: 19,
    )

    args = argparse.Namespace(checkpoint=checkpoint, output=output, opset=18)

    with pytest.raises(RuntimeError, match="Expected ONNX opset 18, got 19"):
        export_corner_cnn_onnx.export_checkpoint(args)
