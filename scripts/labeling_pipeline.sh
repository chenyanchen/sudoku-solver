#!/usr/bin/env bash
# Labeling pipeline orchestrator for YOLO + CNN training data.
#
# Usage:
#   ./scripts/labeling_pipeline.sh generate    # Step 1: YOLO warp + OCR pre-annotate
#   ./scripts/labeling_pipeline.sh upload      # Step 2: Upload to Label Studio
#   # ... manual labeling in Label Studio ...
#   ./scripts/labeling_pipeline.sh export      # Step 3: Export back to repo
#   ./scripts/labeling_pipeline.sh train       # Step 4: Retrain models
#
# Environment variables:
#   LABEL_STUDIO_URL      (default: http://localhost:8080)
#   LABEL_STUDIO_API_KEY  (required for upload/export)
#   CNN_MODEL_PATH        (required)
#   YOLO_MODEL            (default: models/releases/yolo_grid_detect.onnx)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

YOLO_MODEL="${YOLO_MODEL:-models/releases/yolo_grid_detect.onnx}"

case "${1:-help}" in
  generate)
    echo "=== Generating warped grids + OCR pre-annotations ==="
    uv run python scripts/generate_warped_grids.py \
      --yolo-model "$YOLO_MODEL"
    ;;

  upload)
    echo "=== Uploading to Label Studio: keypoints ==="
    uv run python scripts/upload_to_labelstudio.py --project-type keypoints
    echo ""
    echo "=== Uploading to Label Studio: OCR ==="
    uv run python scripts/upload_to_labelstudio.py --project-type ocr
    ;;

  export)
    echo "=== Exporting keypoint annotations ==="
    uv run python scripts/export_from_labelstudio.py --project-type keypoints
    echo ""
    echo "=== Exporting OCR annotations ==="
    uv run python scripts/export_from_labelstudio.py --project-type ocr --merge
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

  full)
    "$0" generate
    echo ""
    "$0" upload
    echo ""
    echo "============================================"
    echo "  Manual labeling step"
    echo "  Open ${LABEL_STUDIO_URL:-http://localhost:8080}"
    echo "  Review and correct annotations in both projects."
    echo "  When done, run: $0 export"
    echo "============================================"
    ;;

  help|*)
    cat <<USAGE
Usage: $0 <command>

Commands:
  generate  Generate warped grids + OCR pre-annotations
  upload    Upload to Label Studio (keypoints + OCR)
  export    Export annotations back to repo (YOLO labels + sudoku_labels.json)
  train     Retrain YOLO + CNN models
  full      generate → upload → (manual labeling) → prompt

Environment variables:
  LABEL_STUDIO_URL      (default: http://localhost:8080)
  LABEL_STUDIO_API_KEY  (required for upload/export)
  CNN_MODEL_PATH        (required)
  YOLO_MODEL            (default: models/releases/yolo_grid_detect.onnx)
USAGE
    ;;
esac
