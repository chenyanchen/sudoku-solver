"""Release a trained model from MLflow to models/releases/.

Usage:
    uv run python scripts/release_model.py cnn --run-id <id> --version <YYYYMMDD.N>
    uv run python scripts/release_model.py corner-cnn --run-id <id> --version <YYYYMMDD.N>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from mlflow.entities import Run
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASES_DIR = REPO_ROOT / "models" / "releases"


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
    extra: dict[str, object] | None = None,
) -> None:
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
        "source_git_commit": run.data.tags.get("git_commit"),
        "released_at_utc": datetime.now(UTC).isoformat(),
    })
    if extra:
        payload.update(extra)
    artifact_path.with_suffix(".meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
def release_cnn(run_id: str, release_version: str) -> int:
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

        # Release gate uses original photos through the production detector chain.
        # A solvable but incorrectly recognized board must not pass.
        subprocess.run([
            sys.executable, str(REPO_ROOT / "scripts" / "eval_cnn_ocr.py"),
            "--cnn-model", str(output_path), "--strict",
            "--image-pattern", "IMG_86*.jpg", "IMG_870[0-8].jpg",
            "sudoku_2.jpg", "sudoku_3.jpg",
        ], cwd=REPO_ROOT, check=True)

        _write_release_metadata(
            artifact_path=output_path,
            model_type="cnn",
            model_version=release_version,
            run=run,
            metric_name="score",
            metric_value=score,
            extra={"training_version": training_version, "preprocessing": "local_adaptive_v1"},
        )
        canonical = RELEASES_DIR / "sudoku_digit_cnn.onnx"
        canonical.unlink(missing_ok=True)
        canonical.symlink_to(output_path.name)

    print(f"  Output:  {output_path}")
    return 0


def release_corner_cnn(run_id: str, release_version: str) -> int:
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
    args = parser.parse_args()

    if args.model == "cnn":
        return release_cnn(args.run_id, args.version)
    else:
        return release_corner_cnn(args.run_id, args.version)


if __name__ == "__main__":
    raise SystemExit(main())
