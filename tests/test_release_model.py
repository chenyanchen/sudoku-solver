from __future__ import annotations

import pytest

from scripts import release_model


def test_main_requires_explicit_run_id(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["release_model.py", "yolo", "--version", "20260331.4"],
    )

    with pytest.raises(SystemExit) as excinfo:
        release_model.main()

    assert excinfo.value.code == 2


def test_main_passes_explicit_run_id_to_yolo_release(monkeypatch):
    captured: dict[str, object] = {}

    def fake_release_yolo(run_id: str, release_version: str) -> int:
        captured["run_id"] = run_id
        captured["release_version"] = release_version
        return 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "release_model.py",
            "yolo",
            "--version",
            "20260331.4",
            "--run-id",
            "run-123",
        ],
    )
    monkeypatch.setattr(release_model, "release_yolo", fake_release_yolo)

    assert release_model.main() == 0
    assert captured == {"run_id": "run-123", "release_version": "20260331.4"}
