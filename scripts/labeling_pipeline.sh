#!/usr/bin/env bash
# Labeling pipeline orchestrator for YOLO + CNN training data.
#
# Usage:
#   ./scripts/labeling_pipeline.sh backend     # Start ML Backend (port 9090)
#   ./scripts/labeling_pipeline.sh generate    # Generate warped grids for OCR project
#   ./scripts/labeling_pipeline.sh train       # Retrain YOLO + CNN models
#   ./scripts/labeling_pipeline.sh release     # Export a versioned release from MLflow
#
# Environment variables:
#   LABEL_STUDIO_URL      (default: http://localhost:8080)
#   LABEL_STUDIO_API_KEY  (required for ML Backend to fetch images)
#   CNN_MODEL_PATH        (required)
#   YOLO_MODEL            (required)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

case "${1:-help}" in
  backend)
    : "${CNN_MODEL_PATH:?CNN_MODEL_PATH must point to an explicit released CNN ONNX}"
    : "${YOLO_MODEL:?YOLO_MODEL must point to an explicit released YOLO ONNX}"
    echo "=== Starting ML Backend on port ${2:-9090} ==="
    uv run python -m ml_backend --port "${2:-9090}"
    ;;

  generate)
    echo "=== Generating warped grids + OCR pre-annotations ==="
    uv run python scripts/generate_warped_grids.py \
      --yolo-model "${YOLO_MODEL:-models/releases/yolo_grid_detect.onnx}"
    ;;

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

  release)
    export MLFLOW_TRACKING_URI="file://${REPO_ROOT}/mlruns"
    shift
    uv run python scripts/release_model.py "$@"
    ;;

  help|*)
    cat <<USAGE
Usage: $0 <command>

Commands:
  backend [port]  Start ML Backend server (default: 9090)
  generate        Generate warped grids for OCR project import
  train           Retrain YOLO + CNN models (metrics logged to MLflow)
  release <cnn|yolo> --version <YYYYMMDD.N> [--run-id <id>]  Export versioned model to models/releases/

Workflow:
  1. Release explicit models:
     $0 release cnn --version 20260331.1
     $0 release yolo --version 20260331.1
  2. Start ML Backend:
     CNN_MODEL_PATH=models/releases/sudoku_digit_cnn.onnx \\
     YOLO_MODEL=models/releases/yolo_grid_detect.onnx \\
     $0 backend
  3. Connect in Label Studio: Settings → Model → http://localhost:9090
  4. Import images into projects (Label Studio UI)
  5. Label (ML Backend auto-predicts + /fit auto-saves corrections)
  6. Train: $0 train
  7. Review metrics: mlflow ui --port 5000 --backend-store-uri file://\$(pwd)/mlruns

Environment variables:
  LABEL_STUDIO_URL      (default: http://localhost:8080)
  LABEL_STUDIO_API_KEY  (required for ML Backend to fetch images)
  CNN_MODEL_PATH        (required)
  YOLO_MODEL            (required)
  MLFLOW_TRACKING_URI   (default: file://\$REPO_ROOT/mlruns, set by train/release)
USAGE
    ;;
esac
