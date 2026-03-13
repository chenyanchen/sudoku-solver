# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based Sudoku solver with three interfaces: a FastAPI web app (image upload), a macOS screen overlay (real-time detection), and a camera monitor prototype (YOLO-based). The pipeline is: **detect grid (OpenCV) → extract cells → OCR digits (CNN/ONNX) → solve (backtracking) → display**.

## Commands

```bash
# Install & sync
uv sync                    # core deps (FastAPI, OpenCV, onnxruntime)
uv sync --extra dev        # + pytest
uv sync --extra ml         # + PyTorch, ultralytics (training only)
uv sync --extra screen     # + PyQt6, mss (macOS overlay)

# Run web server
uv run uvicorn backend.main:app --reload

# Run screen monitor
CNN_MODEL_PATH=models/releases/sudoku_digit_cnn_v3.2.onnx uv run python -m screen_monitor --debug

# Tests
uv run pytest              # all tests
uv run pytest tests/test_solver.py             # single file
uv run pytest tests/test_cv.py -k "test_name"  # single test

# CNN training workflow
uv run python scripts/train_cnn_ocr.py --labels data/labels/sudoku_labels.json --output models/checkpoints/sudoku_digit_cnn.pt --version 3.2.0
uv run python scripts/export_cnn_onnx.py --checkpoint models/checkpoints/sudoku_digit_cnn.pt --output models/releases/sudoku_digit_cnn_v3.2.onnx
uv run python scripts/eval_cnn_ocr.py --labels data/labels/sudoku_labels.json --cnn-model models/releases/sudoku_digit_cnn_v3.2.onnx
```

## Architecture

### Processing Pipeline (shared by web app and screen monitor)

1. **Grid detection** (`backend/cv/grid_detector.py`): OpenCV contour detection → quadrilateral candidates → perspective warp to 450×450. Handles decorative borders via grid-line regularity scoring and line-based refinement.
2. **Cell extraction** (`backend/cv/cell_extractor.py`): Splits warped grid into 81 cells.
3. **OCR** (`backend/ocr/cnn_digit_reader.py`): Custom CNN (48×48 grayscale, 10 classes), runs via ONNX Runtime. Three-phase recognition: blank detection → batch inference on 81 cells → uncertain-cell recheck with adaptive threshold variant. Applies Sudoku-constraint reranking on low-confidence cells.
4. **Grid repair** (`backend/ocr/grid_repair.py`): When OCR produces an unsolvable grid, bounded search over CNN candidate alternatives (configurable max_changes budget) to find a solvable correction.
5. **Solver** (`backend/solver/backtracking.py`): Backtracking with MRV heuristic (picks cell with fewest legal candidates). 100K iteration limit.

### Web App (`backend/`)

FastAPI app at `backend/main.py`. CNN model is validated at startup (lifespan hook fails fast on missing model). OCR thresholds configurable via environment variables (`CNN_BLANK_THRESHOLD`, `CNN_DIGIT_THRESHOLD`, etc.).

### Screen Monitor (`screen_monitor/`)

macOS overlay daemon. Runs as `python -m screen_monitor`. Architecture:

- **agent.py**: Main coordination — capture loop in daemon thread, Qt app on main thread. Two-layer cache (visual hash → puzzle hash) avoids redundant OCR/solve. ROI-based detection uses frame diff regions + last known grid bbox.
- **solver_pipeline.py**: Screen-specific OCR+solve path (skips CLAHE since screen has uniform lighting).
- **overlay.py / hud_painter.py**: PyQt6 transparent overlay, click-through. Debug HUD shows telemetry.
- **grid_tracker.py**: Stability tracking — requires N consecutive stable frames before showing overlay.

### CNN Model Architecture (`backend/ocr/model_arch.py`)

Three architectures: `custom_small_cnn` (v1-v2), `custom_cnn_v3` (ResBlock + SE attention, current), `timm_mobilenetv3`. Training requires `--extra ml`.

### Camera Monitor (`camera_monitor/`)

Prototype for camera-based (phone photo / Android / wearable) grid detection. **Must use YOLO26n-pose** — do not use Claude Vision API or other alternatives for this path. YOLO model: `models/releases/yolo_grid_detect.onnx`. Validation baseline: `data/raw/images/photo_1.png` must be correctly recognized.

### Key Design Decisions

- **Detection and OCR are decoupled**: detect on downscaled frame (max 1280px width), re-warp 450×450 from original full-res frame for OCR quality.
- **Screen monitor uses OpenCV contours** (sufficient for screenshots); **camera/photo uses YOLO26n-pose** (required for complex backgrounds, lighting, tilt).
- **onnxruntime is a core runtime dependency** (not just ML extra). The backend won't start without it.
- **Screen monitor vs web app preprocessing differ**: screen monitor skips CLAHE (uniform lighting), web app applies CLAHE for camera photos.
- Model files live in `models/releases/` (ONNX, committed) and `models/checkpoints/` (PyTorch, gitignored).

## Environment Variables

Key runtime configuration (all optional, have defaults):

- `CNN_MODEL_PATH` — path to ONNX model (default: `models/releases/sudoku_digit_cnn_v3.2.onnx`)
- `CNN_BLANK_THRESHOLD`, `CNN_DIGIT_THRESHOLD`, `CNN_RERANK_CONFIDENCE`, `CNN_TOPK_CANDIDATES`
- `CNN_REPAIR_MAX_CHANGES`, `CNN_REPAIR_MAX_CELLS`
- `OCR_ENGINE` — only `cnn` is supported (others fallback to cnn)
