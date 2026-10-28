"""MLflow tracking in the repo's mlruns/, recording no absolute paths."""

from __future__ import annotations

import re
from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = "mlruns"


def use_repo_mlruns(experiment: str | None = None) -> None:
    """Track runs in mlruns/ under the repo root, and select `experiment` if given.

    MLflow's file store records each experiment's artifact location as an
    absolute path and every run inherits it, so after the repo moved, new runs
    wrote their artifacts to the old directory. A location relative to the
    repo root survives a move, but MLflow resolves it against the working
    directory, so the scripts must run from the repo root.
    """
    if Path.cwd().resolve() != REPO_ROOT:
        raise SystemExit(f"Run from the repo root ({REPO_ROOT}): MLflow resolves paths in mlruns/ against it.")
    mlflow.set_tracking_uri(MLRUNS)
    if experiment is None:
        return
    mlflow.set_experiment(experiment)
    # New experiments (and the store's Default) are created with an absolute
    # location; runs take theirs from these files.
    absolute = re.compile(rf"^artifact_location: (file://)?{re.escape(str(REPO_ROOT))}/", re.MULTILINE)
    for meta in (REPO_ROOT / MLRUNS).glob("*/meta.yaml"):
        text = meta.read_text(encoding="utf-8")
        relative = absolute.sub("artifact_location: ", text)
        if relative != text:
            meta.write_text(relative, encoding="utf-8")
