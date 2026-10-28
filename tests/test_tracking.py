"""mlruns/ records no absolute paths, so runs survive moving the repo."""

import mlflow
import pytest

from scripts import _tracking


@pytest.fixture
def repo(tmp_path, monkeypatch):
    previous = mlflow.get_tracking_uri()
    root = (tmp_path / "repo").resolve()
    root.mkdir()
    monkeypatch.setattr(_tracking, "REPO_ROOT", root)
    monkeypatch.chdir(root)
    yield root
    mlflow.set_tracking_uri(previous)


def test_runs_survive_moving_the_repo(repo, monkeypatch):
    _tracking.use_repo_mlruns("experiment")
    (repo / "model.pt").write_text("weights")
    with mlflow.start_run() as run:
        mlflow.log_artifact(str(repo / "model.pt"))

    for meta in (repo / "mlruns").rglob("meta.yaml"):
        assert str(repo) not in meta.read_text(), meta

    moved = repo.rename(repo.parent / "moved")
    monkeypatch.setattr(_tracking, "REPO_ROOT", moved)
    monkeypatch.chdir(moved)
    _tracking.use_repo_mlruns()
    path = mlflow.MlflowClient().download_artifacts(run.info.run_id, "model.pt", str(moved))
    assert open(path).read() == "weights"


def test_refuses_to_run_outside_the_repo_root(repo, monkeypatch):
    monkeypatch.chdir(repo.parent)
    with pytest.raises(SystemExit):
        _tracking.use_repo_mlruns("experiment")
