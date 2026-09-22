from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import release_model


def test_main_requires_explicit_run_id(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["release_model.py", "cnn", "--version", "20260331.4"],
    )

    with pytest.raises(SystemExit) as excinfo:
        release_model.main()

    assert excinfo.value.code == 2


def test_main_passes_explicit_run_id_to_cnn_release(monkeypatch):
    captured: dict[str, object] = {}

    def fake_release_cnn(run_id: str, release_version: str) -> int:
        captured["run_id"] = run_id
        captured["release_version"] = release_version
        return 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "release_model.py",
            "cnn",
            "--version",
            "20260331.4",
            "--run-id",
            "run-123",
        ],
    )
    monkeypatch.setattr(release_model, "release_cnn", fake_release_cnn)

    assert release_model.main() == 0
    assert captured == {"run_id": "run-123", "release_version": "20260331.4"}


@pytest.mark.parametrize("gate_passes", [False, True])
def test_cnn_release_checks_photos_before_changing_canonical(tmp_path, monkeypatch, gate_passes):
    canonical = tmp_path / "sudoku_digit_cnn.onnx"
    old = tmp_path / "previous.onnx"
    old.write_bytes(b"old")
    canonical.symlink_to(old.name)
    artifact = tmp_path / "sudoku_digit_cnn_20260922.1.onnx"
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run", experiment_id="exp"),
        data=SimpleNamespace(params={"version": "3.5.2"}, tags={}),
    )
    client = SimpleNamespace(
        get_run=lambda _: run,
        get_metric_history=lambda *_: [SimpleNamespace(value=1.0)],
        list_artifacts=lambda _: [SimpleNamespace(path="model.pt")],
        download_artifacts=lambda *_: "model.pt",
    )
    monkeypatch.setattr(release_model, "MlflowClient", lambda: client)
    monkeypatch.setattr(release_model, "RELEASES_DIR", tmp_path)

    def execute(command, **kwargs):
        if "--checkpoint" in command:
            artifact.write_bytes(b"new")
            artifact.with_suffix(".meta.json").write_text(json.dumps({"class_values": list(range(10)), "input_size": 48}))
        else:
            assert "--strict" in command
            assert canonical.resolve() == old
            if not gate_passes:
                raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(release_model.subprocess, "run", execute)
    if not gate_passes:
        with pytest.raises(subprocess.CalledProcessError):
            release_model.release_cnn("run", "20260922.1")
        assert canonical.resolve() == old
    else:
        release_model.release_cnn("run", "20260922.1")
        assert canonical.resolve() == artifact
        metadata = json.loads(artifact.with_suffix(".meta.json").read_text())
        assert metadata["class_values"] == list(range(10))
        assert metadata["model_version"] == "20260922.1"
        assert metadata["source_metric_value"] == 1.0
