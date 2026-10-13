# CLAUDE.md

Sudoku solver with four front ends sharing one recognition pipeline: a FastAPI web app (image upload), a macOS screen overlay, a camera monitor prototype, and a Label Studio ML Backend. Python 3.13, `uv`.

**detect grid → warp 450×450 → `backend/ocr/pipeline.recognize()` (OCR → decode → unique solution) → display**

## Commands

```bash
uv sync --extra dev        # + pytest; other extras: ml (training), screen (overlay), viz (notebooks)
export CNN_MODEL_PATH=models/releases/sudoku_digit_cnn.onnx   # required by app, monitors and tests

uv run uvicorn backend.main:app --reload
uv run --extra screen python -m screen_monitor --debug
CORNER_CNN_MODEL=models/releases/sudoku_corner_cnn.onnx uv run python -m ml_backend --port 9090

uv run pytest                                   # all tests
uv run pytest tests/test_pipeline.py -k name    # one test
```

### OCR model: train → gate → release

```bash
MLFLOW_TRACKING_URI="file://$PWD/mlruns" uv run --extra ml python scripts/train_cnn_ocr.py --version <MAJOR.MINOR.PATCH> --epochs 60 --patience 20
uv run --extra ml python scripts/export_cnn_onnx.py --checkpoint <run .pt> --output <scratch>.onnx
uv run python scripts/eval_cnn_ocr.py --cnn-model <candidate.onnx> --strict --image-pattern 'IMG_86*.jpg' 'IMG_870[0-8].jpg'
uv run python scripts/eval_cnn_ocr.py --cnn-model <candidate.onnx>      # all labeled images
MLFLOW_TRACKING_URI="file://$PWD/mlruns" uv run --extra ml python scripts/release_model.py cnn --run-id <id> --version <YYYYMMDD.N>
```

Release only if the candidate is no worse than the current release on both evals and `pytest` passes; the validation score saturates at 1.0 and is not a release signal. Training version: MAJOR = architecture, MINOR = data/hyperparameters. Release version: `YYYYMMDD.N`; `models/releases/sudoku_digit_cnn.onnx` is a symlink to the current release, with a `.meta.json` sidecar (version, class order).

## Architecture

- **Detection**: web uses Corner CNN (`backend/cv/corner_cnn_detector.py`) with OpenCV contours (`backend/cv/grid_detector.py`) as fallback; screen monitor uses OpenCV only. Detection runs on a ≤1280px-wide frame; OCR re-warps 450×450 from the full-res frame.
- **Pipeline** (`backend/ocr/pipeline.py`), the only OCR → solve path, used by every front end:
  - `prepare_grid_for_ocr`: adaptive-threshold binarization, the same as training. Per-source gray/CLAHE variants measured far worse.
  - `ocr_cells`: 1px-margin crops; zooms in when the grid's median digit height is below the training range (small-font screenshots) and lowers the digit threshold for that grid. Training and eval use the same function.
  - `decode`: a result is trusted only when the givens have exactly one solution. Otherwise it searches ≤2 edits over CNN candidates (fill missed givens if ambiguous, replace misreads if unsolvable).
- **OCR** (`backend/ocr/cnn_digit_reader.py`): CNN on 48×48 cells via ONNX Runtime (`custom_cnn_v3` in `model_arch.py`). Raises at construction if the model can't load.
- **Solver** (`backend/solver/backtracking.py`): bitmask backtracking with MRV, 100K-node budget; `solve_unique` returns a solution only if it is unique.
- **Screen monitor** (`screen_monitor/agent.py`): capture loop in a daemon thread, Qt overlay on the main thread. Detect/OCR only when the frame changed or a rescan is due; changed regions + last grid bbox as ROIs; OCR results cached by the warped grid's visual hash. Requires N stable frames before showing hints. OpenTelemetry → Prometheus, dashboards in `grafana/`.
- **Camera monitor** (`camera_monitor/`): prototype for camera-based (phone photo / Android / wearable) grid detection. **Must use YOLO26n-pose** — do not use Claude Vision API or other alternatives for this path. YOLO model: `models/releases/yolo_grid_detect.onnx`. Validation baseline: `data/raw/images/photo_1.jpg` must be correctly recognized.
- **ML Backend** (`ml_backend/`): Label Studio protocol (`/health`, `/setup`, `/predict`, `/fit`) for corner keypoint and OCR projects. Requires `CNN_MODEL_PATH` and `CORNER_CNN_MODEL` with `.meta.json` sidecars; reports model version `<cnn>+<corner>`. Label Studio bug [#7151](https://github.com/HumanSignal/label-studio/issues/7151): the "Prediction score" column needs `project.model_version` patched to that exact value:
  ```bash
  curl -X PATCH "http://localhost:9111/api/projects/<PROJECT_ID>" -H "Authorization: Token <TOKEN>" \
    -H "Content-Type: application/json" -d '{"model_version": "<VERSION_FROM_/setup>"}'
  ```

## Data

- `data/labels/sudoku_labels.json`: ground-truth grids. Every label must have exactly one solution (the solver rejected two wrong labels).
- `data/labels/sudoku_corners.csv`: reviewed grid corners. Required for every training image; never train on predicted crops.
- `data/raw/images/`: source screenshots and photos. `data/corner_cnn/`: Corner CNN training data. `data/yolo_grid/`: legacy YOLO keypoint data.

## Environment Variables

- `CNN_MODEL_PATH`: **required** for the web app (startup fails fast) and tests; the monitors' `--model` defaults to it.
- `CNN_BLANK_THRESHOLD` (0.65), `CNN_DIGIT_THRESHOLD` (0.55), `CNN_TOPK_CANDIDATES` (4): optional web overrides.
- `CORNER_CNN_MODEL_PATH`: web Corner CNN (defaults to the release symlink). `CORNER_CNN_MODEL`: ML Backend (required).
