"""Git and data provenance shared by training and release scripts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def git_state(allow_dirty: bool) -> tuple[dict[str, str], str]:
    """Return (tags, diff) for HEAD plus uncommitted changes to tracked files.

    A dirty tree is refused unless allow_dirty: a model must be reproducible
    from the commit it records.
    """
    diff = _git("diff", "HEAD")
    if diff and not allow_dirty:
        raise SystemExit(
            "Uncommitted changes to tracked files; commit them or pass --allow-dirty "
            "(the diff is then recorded with the run)."
        )
    tags = {"git_commit": _git("rev-parse", "HEAD").strip(), "git_dirty": str(bool(diff)).lower()}
    if diff:
        tags["git_diff_sha256"] = hashlib.sha256(diff.encode()).hexdigest()
    return tags, diff


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
