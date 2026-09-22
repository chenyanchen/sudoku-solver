#!/usr/bin/env bash
# Labeling pipeline orchestrator for Corner CNN + OCR CNN training data.
#
# Usage:
#   ./scripts/labeling_pipeline.sh backend     # Start ML Backend (port 9090)
#   ./scripts/labeling_pipeline.sh generate    # Generate warped grids for OCR project
#   ./scripts/labeling_pipeline.sh train       # Retrain Corner CNN + OCR CNN models
#   ./scripts/labeling_pipeline.sh release     # Export a versioned release from MLflow
#
# Environment variables:
#   CNN_MODEL_PATH        (required for backend)
#   CORNER_CNN_MODEL      (required for backend)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

case "${1:-help}" in
  backend)
    : "${CNN_MODEL_PATH:?CNN_MODEL_PATH must point to an explicit released CNN ONNX}"
    : "${CORNER_CNN_MODEL:?CORNER_CNN_MODEL must point to an explicit released corner CNN ONNX}"
    echo "=== Starting ML Backend on port ${2:-9090} ==="
    uv run python -m ml_backend --port "${2:-9090}"
    ;;

  generate)
    echo "=== Generating warped grids + OCR pre-annotations ==="
    uv run python scripts/generate_warped_grids.py
    ;;

  train)
    export MLFLOW_TRACKING_URI="file://${REPO_ROOT}/mlruns"

    echo "=== Training Corner CNN ==="
    uv run python scripts/train_corner_cnn.py --epochs 120 --patience 25 --version 20260402.3

    echo ""
    echo "=== Training CNN OCR ==="
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
  train           Retrain Corner CNN + OCR CNN models (metrics logged to MLflow)
  release <cnn|corner-cnn> --version <YYYYMMDD.N> --run-id <id>  Export versioned model

Workflow:
  1. Release explicit models:
     $0 release corner-cnn --version 20260401.1 --run-id <id>
     $0 release cnn --version 20260401.1 --run-id <id>
  2. Start ML Backend:
     CNN_MODEL_PATH=models/releases/sudoku_digit_cnn.onnx \\
     CORNER_CNN_MODEL=models/releases/sudoku_corner_cnn.onnx \\
     $0 backend
  3. Connect in Label Studio: Settings → Model → http://localhost:9090
  4. Import images, label 4 corners (TL, TR, BR, BL)
  5. Train: $0 train
  6. Review metrics: mlflow ui --port 5000 --backend-store-uri file://\$(pwd)/mlruns

Environment variables:
  CNN_MODEL_PATH      (required for backend)
  CORNER_CNN_MODEL    (required for backend)
  MLFLOW_TRACKING_URI (default: file://\$REPO_ROOT/mlruns)
USAGE
    ;;
esac
