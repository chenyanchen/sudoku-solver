# MLflow Training Observability

## Goal

为 CNN 和 YOLO 训练搭建实验追踪系统，支持实时查看 loss/accuracy 曲线、超参数记录、模型产物管理，并建立规范的 release workflow。

## Tool Choice

MLflow 原生集成（文件后端），无需数据库或额外服务。`mlflow ui` 一行启动本地 Web UI。

升级路径：未来需要多人协作时，只需改 `MLFLOW_TRACKING_URI` 指向 MLflow Tracking Server。

## MLflow Concepts Mapping

| MLflow 概念 | 对应本项目 |
|---|---|
| Experiment | `sudoku-cnn-ocr`、`sudoku-yolo-grid` |
| Run | 一次训练脚本执行 |
| Parameters | 超参数（lr、batch_size、epochs、arch、version 等） |
| Metrics | loss、accuracy、F1 等（见下方详细定义） |
| Artifacts | .pt checkpoint |
| Tags | git commit hash |

Storage: `mlruns/` at repo root, added to `.gitignore`.

## CNN Training Integration (`scripts/train_cnn_ocr.py`)

### Parameters (logged once at start)

All training hyperparameters: `lr`, `batch_size`, `epochs`, `arch`, `version`, `weight_decay`, `label_smoothing`, `early_stopping_patience`, `val_split`, `optimizer`, `scheduler`.

Tag: `git_commit` (auto-captured).

### Metrics

| Metric | Granularity | Step meaning |
|---|---|---|
| `train_loss` | Every mini-batch | global_step (cumulative batch count) |
| `val_loss` | Every epoch | epoch number |
| `val_accuracy` | Every epoch | epoch number |
| `val_macro_f1` | Every epoch | epoch number |
| `val_blank_precision` | Every epoch | epoch number |
| `score` | Every epoch | epoch number |
| `lr` | Every epoch | epoch number |
| `f1_digit_1` ... `f1_digit_9` | Every epoch | epoch number |
| `epoch_duration_sec` | Every epoch | epoch number |

### Artifacts (logged at end of training)

- `.pt` checkpoint (best model)
- Confusion matrix image (PNG)

### Removals

Delete JSONL metrics logging (`*.metrics.jsonl` writing logic). MLflow is the single source of truth for metrics. Existing `.metrics.jsonl` files in `models/checkpoints/` can remain but will no longer be produced.

## YOLO Training Integration (`scripts/train_yolo_grid.py`)

Ultralytics has a built-in MLflow callback that activates when `MLFLOW_TRACKING_URI` is set. No changes to the training loop needed.

### Auto-logged by ultralytics

- All hyperparameters
- Per-epoch metrics (train/box_loss, train/pose_loss, val/precision, val/recall, val/mAP etc.)
- Training artifacts (best.pt, last.pt, results.csv, training curve PNGs)

### Our additions

- Set `MLFLOW_EXPERIMENT_NAME=sudoku-yolo-grid` environment variable
- Git commit tag

## Pipeline Changes (`scripts/labeling_pipeline.sh`)

### `train` command (modified)

```bash
train)
    export MLFLOW_TRACKING_URI="file://${REPO_ROOT}/mlruns"

    echo "=== Training YOLO ==="
    export MLFLOW_EXPERIMENT_NAME="sudoku-yolo-grid"
    uv run python scripts/train_yolo_grid.py --epochs 20

    echo "=== Training CNN ==="
    uv run python scripts/train_cnn_ocr.py \
      --labels data/labels/sudoku_labels.json \
      --use-yolo --version 3.4.0
    ;;
```

CNN training script saves `.pt` to a temporary location, uploads to MLflow as artifact, then cleans up. The `--output` flag is removed; MLflow is the sole storage for checkpoints.

Removed: `export_cnn_onnx.py` call (moved to release workflow). `models/checkpoints/` is no longer used.

### `release` command (new)

```bash
# Auto-select best run
./scripts/labeling_pipeline.sh release cnn
./scripts/labeling_pipeline.sh release yolo

# Manual run selection
./scripts/labeling_pipeline.sh release cnn --run-id <run_id>
./scripts/labeling_pipeline.sh release yolo --run-id <run_id>
```

#### CNN release flow

1. Query MLflow experiment `sudoku-cnn-ocr` for best run (by `score` metric), or use `--run-id`
2. Download `.pt` artifact from MLflow to a temp location
3. Run `export_cnn_onnx.py --checkpoint <temp_path> --output models/releases/sudoku_digit_cnn_v<version>.onnx`
4. Print run ID, score, version for human confirmation

#### YOLO release flow

1. Query MLflow experiment `sudoku-yolo-grid` for best run (by `metrics/mAP50(B)` metric), or use `--run-id`
2. Download `best.pt` artifact from MLflow to a temp location
3. Export ONNX: load with ultralytics, `model.export(format="onnx")`
4. Copy to `models/releases/yolo_grid_detect.onnx`
5. Print run ID, mAP, confirmation info

### `export_cnn_onnx.py`

No changes. Continues to accept `--checkpoint <path>` argument. The release workflow downloads `.pt` from MLflow first, then passes the local path.

## File Changes Summary

| File | Change |
|---|---|
| `scripts/train_cnn_ocr.py` | Add MLflow logging (params, step-level train_loss, epoch-level val metrics, artifacts). Remove JSONL writing. |
| `scripts/train_yolo_grid.py` | Set MLflow environment variables. No training loop changes. |
| `scripts/labeling_pipeline.sh` | Set `MLFLOW_TRACKING_URI` in train. Remove export step from train. Add `release` command. |
| `.gitignore` | Add `mlruns/` |

## Unchanged

- `scripts/export_cnn_onnx.py` — interface unchanged, called by release workflow
- `scripts/eval_cnn_ocr.py` — out of scope
- `pyproject.toml` — `mlflow` already in ml extra

## Usage

```bash
# Terminal 1: start MLflow UI
mlflow ui --port 5000 --backend-store-uri file://$(pwd)/mlruns

# Terminal 2: train (metrics appear in real-time)
./scripts/labeling_pipeline.sh train

# Browser: http://localhost:5000

# After reviewing metrics, release the best model
./scripts/labeling_pipeline.sh release cnn
./scripts/labeling_pipeline.sh release yolo
```
