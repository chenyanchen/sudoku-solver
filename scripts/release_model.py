"""Release a trained model from MLflow to models/releases/.

Usage:
    uv run python scripts/release_model.py cnn [--run-id <id>]
    uv run python scripts/release_model.py yolo [--run-id <id>]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASES_DIR = REPO_ROOT / "models" / "releases"


def find_best_run(experiment_name: str, metric: str) -> mlflow.entities.Run:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"Experiment '{experiment_name}' not found in MLflow")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No runs found in experiment '{experiment_name}'")
    return runs[0]


def release_cnn(run_id: str | None) -> int:
    experiment_name = "sudoku-cnn-ocr"

    if run_id:
        client = MlflowClient()
        run = client.get_run(run_id)
    else:
        run = find_best_run(experiment_name, "score")
        run_id = run.info.run_id

    score = run.data.metrics.get("score", 0.0)
    version = run.data.params.get("version", "unknown")

    print(f"CNN Release")
    print(f"  Run ID:  {run_id}")
    print(f"  Score:   {score:.4f}")
    print(f"  Version: {version}")

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
        output_path = RELEASES_DIR / f"sudoku_digit_cnn_v{version}.onnx"
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

        # Update canonical symlink (matches YOLO's yolo_grid_detect.onnx pattern)
        canonical = RELEASES_DIR / "sudoku_digit_cnn.onnx"
        canonical.unlink(missing_ok=True)
        canonical.symlink_to(output_path.name)

    print(f"  Output:  {output_path}")
    return 0


def release_yolo(run_id: str | None) -> int:
    experiment_name = "sudoku-yolo-grid"

    if run_id:
        client = MlflowClient()
        run = client.get_run(run_id)
    else:
        run = find_best_run(experiment_name, "metrics/mAP50(B)")
        run_id = run.info.run_id

    mAP = run.data.metrics.get("metrics/mAP50(B)", 0.0)

    print(f"YOLO Release")
    print(f"  Run ID:  {run_id}")
    print(f"  mAP50:   {mAP:.4f}")

    with tempfile.TemporaryDirectory() as tmpdir:
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        pt_artifacts = [a.path for a in artifacts if a.path.endswith("best.pt")]
        if not pt_artifacts:
            raise SystemExit(f"No best.pt artifact found in run {run_id}")

        local_path = client.download_artifacts(run_id, pt_artifacts[0], tmpdir)
        pt_path = Path(local_path)

        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit("ultralytics required. Run: uv sync --extra ml")

        model = YOLO(str(pt_path))
        onnx_path = model.export(format="onnx", imgsz=640, simplify=True)

        dst = RELEASES_DIR / "yolo_grid_detect.onnx"
        RELEASES_DIR.mkdir(exist_ok=True)
        shutil.copy2(onnx_path, dst)

    print(f"  Output:  {dst}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Release model from MLflow")
    parser.add_argument("model", choices=["cnn", "yolo"], help="Model type to release")
    parser.add_argument("--run-id", default=None, help="MLflow run ID (default: best run)")
    args = parser.parse_args()

    if args.model == "cnn":
        return release_cnn(args.run_id)
    else:
        return release_yolo(args.run_id)


if __name__ == "__main__":
    raise SystemExit(main())
