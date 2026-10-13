"""A model must be reproducible from the commit it records."""

import subprocess

import pytest

from scripts import _provenance


@pytest.fixture
def repo(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "train.py").write_text("lr = 1\n")
    git("add", ".")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    monkeypatch.setattr(_provenance, "REPO_ROOT", tmp_path)
    return tmp_path


def test_clean_tree_records_commit(repo):
    tags, diff = _provenance.git_state(allow_dirty=False)
    assert len(tags["git_commit"]) == 40
    assert tags["git_dirty"] == "false" and diff == ""


def test_dirty_tree_is_refused_unless_allowed(repo):
    (repo / "train.py").write_text("lr = 2\n")

    with pytest.raises(SystemExit, match="--allow-dirty"):
        _provenance.git_state(allow_dirty=False)

    tags, diff = _provenance.git_state(allow_dirty=True)
    assert tags["git_dirty"] == "true"
    assert "+lr = 2" in diff and len(tags["git_diff_sha256"]) == 64
