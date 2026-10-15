from __future__ import annotations

import json
import subprocess
from pathlib import Path
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

    def fake_release_cnn(run_id: str, release_version: str, release_git: dict) -> int:
        captured["run_id"] = run_id
        captured["release_version"] = release_version
        captured["release_git"] = release_git
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
    monkeypatch.setattr(release_model, "git_state", lambda allow_dirty: ({"git_commit": "abc"}, ""))

    assert release_model.main() == 0
    assert captured == {
        "run_id": "run-123", "release_version": "20260331.4", "release_git": {"git_commit": "abc"},
    }


@pytest.mark.parametrize("gate_passes", [False, True])
def test_cnn_release_checks_photos_before_changing_canonical(tmp_path, monkeypatch, gate_passes):
    canonical = tmp_path / "sudoku_digit_cnn.onnx"
    old = tmp_path / "previous.onnx"
    old.write_bytes(b"old")
    canonical.symlink_to(old.name)
    artifact = tmp_path / "sudoku_digit_cnn_20260922.1.onnx"
    run = SimpleNamespace(
        info=SimpleNamespace(run_id="run", experiment_id="exp"),
        data=SimpleNamespace(
            params={"version": "3.5.2", "seed": "42"},
            tags={"git_commit": "c0ffee", "git_dirty": "false", "labels_sha256": "aa", "corners_sha256": "bb"},
        ),
    )
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["a.jpg"], "val": ["b.jpg"]}))
    client = SimpleNamespace(
        get_run=lambda _: run,
        get_metric_history=lambda *_: [SimpleNamespace(value=1.0)],
        list_artifacts=lambda _: [SimpleNamespace(path="model.pt")],
        download_artifacts=lambda _run, path, _dir: str(split) if path == "split.json" else "model.pt",
    )
    monkeypatch.setattr(release_model, "MlflowClient", lambda: client)
    monkeypatch.setattr(release_model, "RELEASES_DIR", tmp_path)

    def execute(command, **kwargs):
        if "--checkpoint" in command:
            artifact.write_bytes(b"new")
            artifact.with_suffix(".meta.json").write_text(json.dumps({"class_values": list(range(10)), "input_size": 48}))
        else:
            assert "--strict" in command and "--image-pattern" not in command  # every label
            assert kwargs["env"]["CNN_MODEL_PATH"] == str(artifact)
            assert canonical.resolve() == old
            if not gate_passes:
                raise subprocess.CalledProcessError(1, command)
            Path(command[command.index("--output-json") + 1]).write_text(json.dumps({"cnn": {
                "pipeline_exact": 16, "solution_exact": 16, "grid_exact": 15,
                "false_positive_blank": 1, "images": 16,
            }}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(release_model.subprocess, "run", execute)
    if not gate_passes:
        with pytest.raises(subprocess.CalledProcessError):
            release_model.release_cnn("run", "20260922.1", {})
        assert canonical.resolve() == old
    else:
        release_model.release_cnn("run", "20260922.1", {"git_commit": "d00d", "git_dirty": "false"})
        assert canonical.resolve() == artifact
        metadata = json.loads(artifact.with_suffix(".meta.json").read_text())
        assert metadata["class_values"] == list(range(10))
        assert metadata["model_version"] == "20260922.1"
        assert metadata["source_metric_value"] == 1.0
        # The sidecar must explain the release without access to mlruns/.
        assert metadata["source_git_commit"] == "c0ffee"
        assert metadata["training_data_sha256"] == {"labels": "aa", "corners": "bb"}
        assert metadata["training_params"]["seed"] == "42"
        assert metadata["val_images"] == ["b.jpg"]
        assert metadata["release_git_commit"] == "d00d"
        assert metadata["release_gate"]["pipeline_exact"] == 16
        assert metadata["release_gate"]["labels"] == "data/labels/sudoku_labels.json"
