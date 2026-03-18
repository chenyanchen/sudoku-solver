#!/usr/bin/env bash
# Labeling pipeline orchestrator for YOLO + CNN training data.
#
# Usage:
#   ./scripts/labeling_pipeline.sh backend     # Start ML Backend (port 9090)
#   ./scripts/labeling_pipeline.sh generate    # Generate warped grids for OCR project
#   ./scripts/labeling_pipeline.sh train       # Retrain YOLO + CNN models
#
# Environment variables:
#   LABEL_STUDIO_URL      (default: http://localhost:8080)
#   LABEL_STUDIO_API_KEY  (required for ML Backend to fetch images)
#   CNN_MODEL_PATH        (required)
#   YOLO_MODEL            (default: models/releases/yolo_grid_detect.onnx)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

case "${1:-help}" in
  backend)
    echo "=== Starting ML Backend on port ${2:-9090} ==="
    uv run python -m ml_backend --port "${2:-9090}"
    ;;

  generate)
    echo "=== Generating warped grids + OCR pre-annotations ==="
    uv run python scripts/generate_warped_grids.py \
      --yolo-model "${YOLO_MODEL:-models/releases/yolo_grid_detect.onnx}"
    ;;

  train)
    echo "=== Training YOLO ==="
    uv run python scripts/train_yolo_grid.py --epochs 20

    echo ""
    echo "=== Training CNN with YOLO warp ==="
    uv run python scripts/train_cnn_ocr.py \
      --labels data/labels/sudoku_labels.json \
      --output models/checkpoints/sudoku_digit_cnn.pt \
      --use-yolo --version 3.4.0
    uv run python scripts/export_cnn_onnx.py \
      --checkpoint models/checkpoints/sudoku_digit_cnn.pt \
      --output models/releases/sudoku_digit_cnn_v3.4.onnx
    ;;

  help|*)
    cat <<USAGE
Usage: $0 <command>

Commands:
  backend [port]  Start ML Backend server (default: 9090)
  generate        Generate warped grids for OCR project import
  train           Retrain YOLO + CNN models

Workflow:
  1. Start ML Backend:  $0 backend
  2. Connect in Label Studio: Settings → Model → http://localhost:9090
  3. Import images into projects (Label Studio UI)
  4. Label (ML Backend auto-predicts + /fit auto-saves corrections)
  5. Train: $0 train

Environment variables:
  LABEL_STUDIO_URL      (default: http://localhost:8080)
  LABEL_STUDIO_API_KEY  (required for ML Backend to fetch images)
  CNN_MODEL_PATH        (required)
  YOLO_MODEL            (default: models/releases/yolo_grid_detect.onnx)
USAGE
    ;;
esac
