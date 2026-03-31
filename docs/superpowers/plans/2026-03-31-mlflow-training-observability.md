# MLflow Training Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate MLflow experiment tracking into CNN and YOLO training with real-time metrics, hyperparameter logging, model artifact management, and a release workflow.

**Architecture:** Training scripts log to MLflow via Python API (file backend at `mlruns/`). CNN gets direct `mlflow.log_*` calls; YOLO uses ultralytics' built-in MLflow callback. A new `release` command queries MLflow for the best run, downloads the checkpoint, and exports to ONNX.

**Tech Stack:** MLflow (already in `pyproject.toml` ml extra), Python, Bash

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `.gitignore` | Modify | Add `mlruns/` |
| `scripts/train_cnn_ocr.py` | Modify | Add MLflow logging, remove JSONL, remove `--output`/`--metrics-log` args, save .pt to temp + upload as artifact |
| `scripts/train_yolo_grid.py` | Modify | Set MLflow env vars, add git commit tag after training |
| `scripts/labeling_pipeline.sh` | Modify | Set `MLFLOW_TRACKING_URI` in train, remove export from train, add `release` command |
| `scripts/release_model.py` | Create | Python script: query MLflow best run, download artifact, call export, print summary |

---

### Task 1: Add `mlruns/` to `.gitignore`

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add mlruns/ to .gitignore**

In `.gitignore`, add under the existing "ML training artifacts" section:

```gitignore
# ML training artifacts
models/checkpoints/
mlruns/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add mlruns/ to .gitignore"
```

---

### Task 2: Integrate MLflow into CNN training script

**Files:**
- Modify: `scripts/train_cnn_ocr.py`

This is the largest task. Changes:
1. Add MLflow experiment/run lifecycle
2. Log params at start
3. Log `train_loss` per mini-batch (step = global_step)
4. Log val metrics per epoch (step = epoch)
5. Log .pt artifact + confusion matrix PNG at end
6. Remove JSONL writing logic
7. Remove `--output` and `--metrics-log` args (MLflow is sole storage)
8. Save .pt to tempfile, upload to MLflow, clean up

- [ ] **Step 1: Add MLflow imports and git commit helper**

At the top of `scripts/train_cnn_ocr.py`, after the existing imports, add:

```python
import subprocess
import tempfile
import time
```

After the `from backend.ocr.model_arch import create_model` line, add:

```python
import mlflow


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"
```

- [ ] **Step 2: Remove `--output` and `--metrics-log` args from `parse_args()`**

In `parse_args()`, remove these two argument definitions:

```python
    # DELETE this block:
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "models" / "checkpoints" / "sudoku_digit_cnn.pt",
        help="Output checkpoint path",
    )
    # DELETE this block:
    parser.add_argument(
        "--metrics-log",
        type=Path,
        default=None,
        help="JSONL path for per-epoch metrics (default: <output>.metrics.jsonl)",
    )
```

- [ ] **Step 3: Add MLflow experiment setup at start of `main()`**

Replace the beginning of `main()` (from `def main()` through the creation of `scheduler`) with the same code but wrapped in MLflow context. After `scheduler = ...`, add MLflow setup:

```python
    mlflow.set_experiment("sudoku-cnn-ocr")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "arch": args.arch,
            "version": args.version,
            "weight_decay": args.weight_decay,
            "label_smoothing": 0.1,
            "early_stopping_patience": args.patience,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "input_size": args.input_size,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "num_train_samples": len(train_samples),
            "num_val_samples": len(val_samples),
            "use_yolo": args.use_yolo,
        })
        mlflow.set_tag("git_commit", _git_commit_hash())
```

The entire training loop and post-training code must be indented inside this `with` block.

- [ ] **Step 4: Add step-level train_loss logging in the training loop**

Replace the inner training loop with a version that tracks `global_step`:

Before the `for epoch` loop, add:

```python
        global_step = 0
```

Inside the mini-batch loop, after `train_loss += float(loss.item())`, add:

```python
            global_step += 1
            mlflow.log_metric("train_loss", loss.item(), step=global_step)
```

- [ ] **Step 5: Replace JSONL writing with MLflow epoch-level metric logging**

Remove the JSONL metrics block (the `metrics_record = {...}` dict and the `with metrics_log_path.open(...)` write). Also remove the `metrics_log_path` variable setup before the epoch loop.

Replace with MLflow logging after the `score = ...` computation:

```python
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_metrics = {
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1_digits"],
            "val_blank_precision": val_metrics["blank_precision"],
            "score": score,
            "lr": current_lr,
            "epoch_duration_sec": epoch_duration,
        }
        for digit, f1_val in val_metrics.get("per_digit_f1", {}).items():
            epoch_metrics[f"f1_digit_{digit}"] = f1_val
        mlflow.log_metrics(epoch_metrics, step=epoch)
```

For `epoch_duration`, add timing around the epoch. Before `model.train()` inside the epoch loop:

```python
        epoch_start = time.time()
```

After `scheduler.step()`:

```python
        epoch_duration = time.time() - epoch_start
```

- [ ] **Step 6: Replace checkpoint saving with MLflow artifact upload**

Remove the old checkpoint save block at the end (the `args.output.parent.mkdir(...)`, `torch.save(...)`, meta JSON writing, and the two print statements).

Replace with:

```python
        if best_state is None:
            raise RuntimeError("No valid checkpoint produced")

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = Path(tmpdir) / f"sudoku_digit_cnn_v{args.version}.pt"
            torch.save(best_state, pt_path)
            mlflow.log_artifact(str(pt_path))

            # Save confusion matrix as PNG artifact
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 8))
                confusion = best_confusion
                im = ax.imshow(confusion, cmap="Blues")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                ax.set_title("Confusion Matrix (best epoch)")
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                for i in range(10):
                    for j in range(10):
                        ax.text(j, i, str(confusion[i][j]), ha="center", va="center",
                                color="white" if confusion[i][j] > max(max(row) for row in confusion) / 2 else "black",
                                fontsize=8)
                fig.colorbar(im)
                fig.tight_layout()
                cm_path = Path(tmpdir) / "confusion_matrix.png"
                fig.savefig(cm_path, dpi=100)
                plt.close(fig)
                mlflow.log_artifact(str(cm_path))
            except ImportError:
                pass  # matplotlib not available, skip confusion matrix image

        print(f"Run ID: {run.info.run_id}")
        print(f"Best score: {best_score:.4f}")
```

To capture `best_confusion`, add inside the `if is_best:` block (alongside `best_state`):

```python
            best_confusion = val_metrics.get("confusion", [])
```

And initialize it before the epoch loop:

```python
        best_confusion: list[list[int]] = []
```

- [ ] **Step 7: Verify the full refactored `main()` function structure**

The final structure of `main()` should be:

```
def main() -> int:
    # torch imports, args, seed, device
    # build_samples, stratified_split, dataloaders
    # model, criterion, optimizer, scheduler

    mlflow.set_experiment("sudoku-cnn-ocr")
    with mlflow.start_run() as run:
        mlflow.log_params({...})
        mlflow.set_tag("git_commit", ...)

        best_score = -1.0
        best_state = None
        best_confusion = []
        epochs_no_improve = 0
        global_step = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0

            for images, labels in train_loader:
                # forward, backward, step
                global_step += 1
                mlflow.log_metric("train_loss", loss.item(), step=global_step)

            train_loss /= max(1, len(train_loader))
            val_metrics = evaluate(...)
            score = ...
            epoch_duration = time.time() - epoch_start

            print(...)  # console output (keep existing)

            mlflow.log_metrics({...}, step=epoch)

            if is_best:
                best_score = score
                best_state = {...}
                best_confusion = val_metrics.get("confusion", [])
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            scheduler.step()

            if epochs_no_improve >= args.patience:
                print(f"Early stop triggered at epoch={epoch}")
                break

        # Upload artifacts
        if best_state is None:
            raise RuntimeError("No valid checkpoint produced")

        with tempfile.TemporaryDirectory() as tmpdir:
            # save .pt, log artifact
            # save confusion matrix PNG, log artifact

        print(f"Run ID: {run.info.run_id}")
        print(f"Best score: {best_score:.4f}")

    return 0
```

- [ ] **Step 8: Run training to verify MLflow integration**

```bash
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
uv run python scripts/train_cnn_ocr.py --labels data/labels/sudoku_labels.json --use-yolo --version test --epochs 3
```

Expected: Training completes, prints Run ID. Then verify:

```bash
uv run mlflow ui --port 5000 --backend-store-uri "file://$(pwd)/mlruns"
```

Open http://localhost:5000, check experiment `sudoku-cnn-ocr` has a run with params, metrics curves, and .pt artifact.

- [ ] **Step 9: Commit**

```bash
git add scripts/train_cnn_ocr.py
git commit -m "feat: integrate MLflow into CNN training, remove JSONL logging"
```

---

### Task 3: Integrate MLflow into YOLO training script

**Files:**
- Modify: `scripts/train_yolo_grid.py`

Ultralytics auto-activates its MLflow callback when `MLFLOW_TRACKING_URI` is set. We just need to set the experiment name and add a git commit tag.

- [ ] **Step 1: Add MLflow env var setup and git tag in `main()`**

At the top of `scripts/train_yolo_grid.py`, add import:

```python
import os
import subprocess
```

In `main()`, right before `model.train(...)`, add:

```python
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "sudoku-yolo-grid")
```

After `model.train(...)` returns, add git commit tag to the run ultralytics created. Note: ultralytics' callback calls `mlflow.end_run()` at the end of training, so we must use the MlflowClient to find and tag the most recent run:

```python
    # Tag the MLflow run with git commit (run is already ended by ultralytics)
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name("sudoku-yolo-grid")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )
            if runs:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip()
                client.set_tag(runs[0].info.run_id, "git_commit", git_hash)
    except Exception:
        pass  # MLflow not configured or git not available
```

- [ ] **Step 2: Test YOLO MLflow integration**

```bash
export MLFLOW_TRACKING_URI="file://$(pwd)/mlruns"
uv run python scripts/train_yolo_grid.py --epochs 1
```

Expected: Training completes. Check MLflow UI for experiment `sudoku-yolo-grid` with metrics and artifacts.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_yolo_grid.py
git commit -m "feat: enable MLflow tracking for YOLO training"
```

---

### Task 4: Update `labeling_pipeline.sh` train command

**Files:**
- Modify: `scripts/labeling_pipeline.sh`

- [ ] **Step 1: Modify the `train)` case**

Replace the existing `train)` block with:

```bash
  train)
    export MLFLOW_TRACKING_URI="file://${REPO_ROOT}/mlruns"

    echo "=== Training YOLO ==="
    export MLFLOW_EXPERIMENT_NAME="sudoku-yolo-grid"
    uv run python scripts/train_yolo_grid.py --epochs 20

    echo ""
    echo "=== Training CNN with YOLO warp ==="
    uv run python scripts/train_cnn_ocr.py \
      --labels data/labels/sudoku_labels.json \
      --use-yolo --version 3.4.0
    ;;
```

Key changes from original:
- Added `MLFLOW_TRACKING_URI` and `MLFLOW_EXPERIMENT_NAME` exports
- Removed `--output` flag from CNN training (no longer exists)
- Removed `export_cnn_onnx.py` call (moved to release workflow)

- [ ] **Step 2: Commit**

```bash
git add scripts/labeling_pipeline.sh
git commit -m "refactor: pipeline train sets MLflow env, removes export step"
```

---

### Task 5: Create release workflow script

**Files:**
- Create: `scripts/release_model.py`

- [ ] **Step 1: Create `scripts/release_model.py`**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add scripts/release_model.py
git commit -m "feat: add release_model.py to publish best MLflow run to models/releases/"
```

---

### Task 6: Add `release` command to `labeling_pipeline.sh`

**Files:**
- Modify: `scripts/labeling_pipeline.sh`

- [ ] **Step 1: Add release case before the help case**

Insert a new case between the `train)` and `help|*)` cases:

```bash
  release)
    export MLFLOW_TRACKING_URI="file://${REPO_ROOT}/mlruns"
    shift
    uv run python scripts/release_model.py "$@"
    ;;
```

- [ ] **Step 2: Update the help text**

Add `release` to the help text:

```bash
  help|*)
    cat <<USAGE
Usage: $0 <command>

Commands:
  backend [port]  Start ML Backend server (default: 9090)
  generate        Generate warped grids for OCR project import
  train           Retrain YOLO + CNN models (metrics logged to MLflow)
  release <cnn|yolo> [--run-id <id>]  Export best model to models/releases/

Workflow:
  1. Start ML Backend:  $0 backend
  2. Connect in Label Studio: Settings → Model → http://localhost:9090
  3. Import images into projects (Label Studio UI)
  4. Label (ML Backend auto-predicts + /fit auto-saves corrections)
  5. Train: $0 train
  6. Review metrics: mlflow ui --port 5000 --backend-store-uri file://\$(pwd)/mlruns
  7. Release: $0 release cnn && $0 release yolo

Environment variables:
  LABEL_STUDIO_URL      (default: http://localhost:8080)
  LABEL_STUDIO_API_KEY  (required for ML Backend to fetch images)
  CNN_MODEL_PATH        (required)
  YOLO_MODEL            (default: models/releases/yolo_grid_detect.onnx)
  MLFLOW_TRACKING_URI   (default: file://\$REPO_ROOT/mlruns, set by train/release)
USAGE
    ;;
```

- [ ] **Step 3: Test the full pipeline**

```bash
# Train (verify MLflow receives data)
./scripts/labeling_pipeline.sh train

# Check MLflow UI
uv run mlflow ui --port 5000 --backend-store-uri "file://$(pwd)/mlruns"

# Release
./scripts/labeling_pipeline.sh release cnn
./scripts/labeling_pipeline.sh release yolo
```

- [ ] **Step 4: Commit**

```bash
git add scripts/labeling_pipeline.sh
git commit -m "feat: add release command to labeling pipeline"
```
