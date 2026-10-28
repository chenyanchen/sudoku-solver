"""Release a trained model from MLflow to models/releases/.

Usage:
    uv run python scripts/release_model.py cnn --run-id <id> --version <YYYYMMDD.N>
    uv run python scripts/release_model.py corner-cnn --run-id <id> --version <YYYYMMDD.N>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._provenance import git_state  # noqa: E402
from scripts._tracking import use_repo_mlruns  # noqa: E402

RELEASES_DIR = REPO_ROOT / "models" / "releases"
# The OCR release gate: every labeled image, read exactly through the production
# pipeline, so each image added to the labels is protected from then on. The test
# set stays out: deciding releases on it would tune the model to it.
CNN_GATE_LABELS = REPO_ROOT / "data" / "labels" / "sudoku_labels.json"


def _validate_release_version(value: str) -> str:
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("release version must be non-empty")
    value = value.strip()
    if not all(ch.isdigit() or ch == "." for ch in value):
        raise argparse.ArgumentTypeError(
            "release version must use the unified numeric format YYYYMMDD.N"
        )
    parts = value.split(".")
    if len(parts) != 2 or len(parts[0]) != 8 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(
            "release version must use the unified numeric format YYYYMMDD.N"
        )
    return value


def _write_release_metadata(
    *,
    artifact_path: Path,
    model_type: str,
    model_version: str,
    run: Run,
    metric_name: str,
    metric_value: float,
    release_git: dict[str, str],
    extra: dict[str, object] | None = None,
) -> None:
    """Write a sidecar that explains the release without access to mlruns/."""
    tags = run.data.tags
    meta_path = artifact_path.with_suffix(".meta.json")
    # Preserve exporter metadata: class order, input size and training version.
    payload: dict[str, object] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    payload.update({
        "model_type": model_type,
        "model_version": model_version,
        "source_run_id": run.info.run_id,
        "source_experiment_id": run.info.experiment_id,
        "source_metric_name": metric_name,
        "source_metric_value": metric_value,
        "source_git_commit": tags.get("git_commit"),
        "source_git_dirty": tags.get("git_dirty"),
        "source_git_diff_sha256": tags.get("git_diff_sha256"),
        "training_data_sha256": {
            key.removesuffix("_sha256"): value
            for key, value in tags.items()
            if key.endswith("_sha256") and key != "git_diff_sha256"
        },
        "training_params": dict(run.data.params),
        "release_git_commit": release_git.get("git_commit"),
        "release_git_dirty": release_git.get("git_dirty"),
        "released_at_utc": datetime.now(UTC).isoformat(),
    })
    if extra:
        payload.update(extra)
    artifact_path.with_suffix(".meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
def release_cnn(run_id: str, release_version: str, release_git: dict[str, str]) -> int:
    client = MlflowClient()
    run = client.get_run(run_id)

    history = client.get_metric_history(run_id, "score")
    score = max((metric.value for metric in history), default=0.0)
    training_version = run.data.params.get("version")

    print("CNN Release")
    print(f"  Run ID:  {run_id}")
    print(f"  Score:   {score:.4f}")
    print(f"  Version: {release_version}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Find and download the .pt artifact
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        pt_artifacts = [a.path for a in artifacts if a.path.endswith(".pt")]
        if not pt_artifacts:
            raise SystemExit(f"No .pt artifact found in run {run_id}")

        local_path = client.download_artifacts(run_id, pt_artifacts[0], tmpdir)
        pt_path = Path(local_path)

        # Run export
        output_path = RELEASES_DIR / f"sudoku_digit_cnn_{release_version}.onnx"
        if output_path.exists():
            raise SystemExit(f"Release already exists: {output_path}")
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "export_cnn_onnx.py"),
                "--checkpoint", str(pt_path),
                "--output", str(output_path),
            ],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            raise SystemExit(f"ONNX export failed with code {result.returncode}")

        # The detector chain reads CNN_MODEL_PATH too, so it must see the candidate.
        gate_json = Path(tmpdir) / "gate.json"
        subprocess.run([
            sys.executable, str(REPO_ROOT / "scripts" / "eval_cnn_ocr.py"),
            "--cnn-model", str(output_path), "--strict",
            "--labels", str(CNN_GATE_LABELS),
            "--output-json", str(gate_json),
        ], cwd=REPO_ROOT, check=True, env={**os.environ, "CNN_MODEL_PATH": str(output_path)})
        gate = json.loads(gate_json.read_text())["cnn"]

        split_path = Path(client.download_artifacts(run_id, "split.json", tmpdir))
        val_images = json.loads(split_path.read_text())["val"]

        _write_release_metadata(
            artifact_path=output_path,
            model_type="cnn",
            model_version=release_version,
            run=run,
            metric_name="score",
            metric_value=score,
            release_git=release_git,
            extra={
                "training_version": training_version,
                "preprocessing": "local_adaptive_v1",
                "val_images": val_images,
                "release_gate": {
                    "labels": str(CNN_GATE_LABELS.relative_to(REPO_ROOT)),
                    **{k: gate[k] for k in ("pipeline_exact", "solution_exact", "grid_exact", "false_positive_blank")},
                    "total": gate["images"],
                },
            },
        )
        canonical = RELEASES_DIR / "sudoku_digit_cnn.onnx"
        canonical.unlink(missing_ok=True)
        canonical.symlink_to(output_path.name)

    print(f"  Output:  {output_path}")
    return 0


def release_corner_cnn(run_id: str, release_version: str, release_git: dict[str, str]) -> int:
    client = MlflowClient()
    run = client.get_run(run_id)

    warp_iou = run.data.metrics.get("warp_iou", 0.0)

    print("Corner CNN Release")
    print(f"  Run ID:    {run_id}")
    print(f"  warp_iou:  {warp_iou:.4f}")
    print(f"  Version:   {release_version}")

    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = client.list_artifacts(run_id)
        pt_artifacts = [a.path for a in artifacts if a.path.endswith(".pt")]
        if not pt_artifacts:
            raise SystemExit(f"No .pt artifact found in run {run_id}")

        local_path = client.download_artifacts(run_id, pt_artifacts[0], tmpdir)
        pt_path = Path(local_path)

        output_path = RELEASES_DIR / f"sudoku_corner_cnn_{release_version}.onnx"
        result = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / "scripts" / "export_corner_cnn_onnx.py"),
                "--checkpoint", str(pt_path),
                "--output", str(output_path),
            ],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            raise SystemExit(f"ONNX export failed with code {result.returncode}")

        canonical = RELEASES_DIR / "sudoku_corner_cnn.onnx"
        canonical.unlink(missing_ok=True)
        canonical.symlink_to(output_path.name)

        _write_release_metadata(
            artifact_path=output_path,
            model_type="corner_cnn",
            model_version=release_version,
            run=run,
            metric_name="warp_iou",
            metric_value=warp_iou,
            release_git=release_git,
        )

    print(f"  Output:  {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Release model from MLflow")
    parser.add_argument("model", choices=["cnn", "corner-cnn"], help="Model type to release")
    parser.add_argument(
        "--version",
        required=True,
        type=_validate_release_version,
        help="Unified release version in YYYYMMDD.N format",
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID to release")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="Release with uncommitted changes (the gate then ran on uncommitted code)",
    )
    args = parser.parse_args()

    use_repo_mlruns()
    release_git, _ = git_state(args.allow_dirty)
    if args.model == "cnn":
        return release_cnn(args.run_id, args.version, release_git)
    return release_corner_cnn(args.run_id, args.version, release_git)


if __name__ == "__main__":
    raise SystemExit(main())
