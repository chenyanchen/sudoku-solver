# CLAUDE.md

Sudoku solver with four front ends sharing one recognition pipeline: a FastAPI web app (image upload), a macOS screen overlay, a camera monitor prototype, and a Label Studio ML Backend. Python 3.13, `uv`.

**detect grid → warp 450×450 → `backend/ocr/pipeline.recognize()` (OCR → decode → unique solution) → display**

## Commands

```bash
uv sync --extra dev        # + pytest (ML tests are skipped without --extra ml); other extras: ml (training), labeling (ML Backend), screen (overlay), viz (notebooks)
export CNN_MODEL_PATH=models/releases/sudoku_digit_cnn.onnx   # required by app, monitors and tests

uv run uvicorn backend.main:app --reload
uv run --extra screen python -m screen_monitor --debug
CORNER_CNN_MODEL=models/releases/sudoku_corner_cnn.onnx uv run --extra labeling python -m ml_backend --port 9090

uv run pytest                                   # all tests
uv run pytest tests/test_pipeline.py -k name    # one test
```

### Corner CNN: train → eval → release

```bash
uv run --extra ml python scripts/synth_corner_data.py --count 6000 --seed 0          # → gitignored data/synth_corner/
MLFLOW_TRACKING_URI="file://$PWD/mlruns" uv run --extra ml python scripts/train_corner_cnn.py --synth data/synth_corner --version <MAJOR.MINOR.PATCH>
uv run --extra ml python scripts/export_corner_cnn_onnx.py --checkpoint <run .pt> --output <scratch>.onnx
uv run python scripts/eval_corner_cnn.py --model <candidate.onnx>      # held-out test set; compare with the release
MLFLOW_TRACKING_URI="file://$PWD/mlruns" uv run --extra ml python scripts/release_model.py corner-cnn --run-id <id> --version <YYYYMMDD.N>
```

Pick between candidates on training/val error (`eval_corner_cnn.py --labels data/labels/sudoku_labels.json --corners data/labels/sudoku_corners.csv`), not the test set. Synthetic images are training-only; selection uses real val images. Regressing corners from globally pooled features (1.x) underfit: 4–9% median error even on training images; heatmaps (2.x) fixed that.

### OCR model: train → gate → release

```bash
MLFLOW_TRACKING_URI="file://$PWD/mlruns" uv run --extra ml python scripts/train_cnn_ocr.py --version <MAJOR.MINOR.PATCH> --epochs 60 --patience 20
uv run --extra ml python scripts/export_cnn_onnx.py --checkpoint <run .pt> --output <scratch>.onnx
uv run python scripts/eval_cnn_ocr.py --cnn-model <candidate.onnx> --strict --image-pattern 'IMG_86*.jpg' 'IMG_870[0-8].jpg'
uv run python scripts/eval_cnn_ocr.py --cnn-model <candidate.onnx>      # all labeled images
MLFLOW_TRACKING_URI="file://$PWD/mlruns" uv run --extra ml python scripts/release_model.py cnn --run-id <id> --version <YYYYMMDD.N>
```

Release only if the candidate is no worse than the current release on both evals and `pytest` passes; the validation score saturates at 1.0 and is not a release signal. Seeds alone swing results (the same architecture scored 59–61/61), so compare architectures over 3 seeds each. Training uses MPS when available (separable convs train ~50× slower on CPU); the device changes results, so compare runs on one device. Training and release refuse uncommitted changes (`--allow-dirty` records the diff), so the sidecar's commit reproduces the model; the sidecar also records data hashes, params, the val split and gate results, so it stands alone without `mlruns/`. Training version: MAJOR = architecture, MINOR = data/hyperparameters. Release version: `YYYYMMDD.N`; `models/releases/sudoku_digit_cnn.onnx` is a symlink to the current release, with a `.meta.json` sidecar (version, class order).

### Deploy (public site)

```bash
scp -r deploy <host>:/root/sudoku-deploy && ssh <host> bash /root/sudoku-deploy/setup-server.sh [domain]   # once; the domain enables Caddy/TLS
deploy/deploy.sh <host>                                  # ships the committed HEAD; keeps 3 releases
```

The site runs on one 2-core/2 GB VPS: uvicorn (1 worker, 127.0.0.1:8000, systemd `sudoku-solver`) behind Caddy (TLS). `SUDOKU_DATA_DIR=/var/lib/sudoku-solver` holds `records.sqlite3` (every upload's result, corrections) and `images/` (every upload, once per SHA-256, 5 GB budget). No access log: the privacy note promises no IPs are stored.

## Architecture

- **Detection**: web (`routes.read_image`) reads the grid from OpenCV contours (`backend/cv/grid_detector.py`) and, when that grid is not a unique reading, from Corner CNN's corners (`backend/cv/corner_cnn_detector.py`; heatmap + soft-argmax, median corner error 1% on the test set). Each solves images the other cannot (glare over the lines, a plastic sleeve / grids Corner CNN mislocates). The unique solution is the acceptance test, so the Corner CNN retry skips the regularity check, which rejects good corners on photos with moiré. Corner CNN also pre-annotates corners in the ML Backend; the screen monitor uses OpenCV only. OpenCV's fixed-pixel thresholds suit no single scale (12MP photos of screens pick up moiré; some paper photos lose lines when downscaled; in images narrower than ~1000px the opening erases 1px lines), so web tries full res then a 1280px-wide copy (down- or upscaled) and takes the first warp with regular grid lines (`has_regular_grid_lines`: lines at the warp's 10 equally spaced positions per axis, so a frame around the grid fails); the screen monitor detects at ≤1280px only. A quad without regular lines is usually a decorative frame or a single 3×3 box (when the grid outline merges with its surroundings); `snap_to_lattice` fits the 9×9 line lattice around it. OCR always re-warps 450×450 from the full-res frame.
- **Pipeline** (`backend/ocr/pipeline.py`), the only OCR → solve path, used by every front end:
  - `prepare_grid_for_ocr`: adaptive-threshold binarization, the same as training. Per-source gray/CLAHE variants measured far worse.
  - `ocr_cells`: 1px-margin crops; zooms in when the grid's median digit height is below the training range (small-font screenshots) and lowers the digit threshold for that grid. Training and eval use the same function.
  - `decode`: a result is trusted only when the givens have exactly one solution. Otherwise it searches ≤2 edits over CNN candidates (fill missed givens if ambiguous, replace misreads if unsolvable). Cells read as blank with blank probability < 0.05 (an unfamiliar font) start as givens with their top candidate, outside that budget.
- **OCR** (`backend/ocr/cnn_digit_reader.py`): CNN on 48×48 cells via ONNX Runtime (`tiny_dsc_16-p-32-32-p-64-64-p-128-128` in `model_arch.py`: 38K params, 8 ms per 81 cells on the VPS; `custom_cnn_v3` took 67 ms at the same accuracy). Raises at construction if the model can't load.
- **Solver** (`backend/solver/backtracking.py`): bitmask backtracking with MRV, 100K-node budget; `solve_unique` returns a solution only if it is unique.
- **Screen monitor** (`screen_monitor/agent.py`): capture loop in a daemon thread, Qt overlay on the main thread. Detect/OCR only when the frame changed or a rescan is due; changed regions + last grid bbox as ROIs; OCR results cached by the warped grid's visual hash. Requires N stable frames before showing hints. OpenTelemetry → Prometheus, dashboards in `grafana/`.
- **Camera monitor** (`camera_monitor/`): prototype for live camera frames. Its YOLO detector's model (`yolo_grid_detect.onnx`) was removed in 90dcddd, so it does not run as shipped.
- **ML Backend** (`ml_backend/`): Label Studio protocol (`/health`, `/setup`, `/predict`, `/fit`) for corner keypoint and OCR projects. Requires `CNN_MODEL_PATH` and `CORNER_CNN_MODEL` with `.meta.json` sidecars; reports model version `<cnn>+<corner>`. Label Studio bug [#7151](https://github.com/HumanSignal/label-studio/issues/7151): the "Prediction score" column needs `project.model_version` patched to that exact value:
  ```bash
  curl -X PATCH "http://localhost:9111/api/projects/<PROJECT_ID>" -H "Authorization: Token <TOKEN>" \
    -H "Content-Type: application/json" -d '{"model_version": "<VERSION_FROM_/setup>"}'
  ```

## Data

- `data/labels/sudoku_labels.json`: ground-truth grids, the digits as printed. Every label must have at least one solution: zero means a transcription error (the solver caught two). Several solutions are allowed when the puzzle itself is ambiguous; check such a label against the image, since a missed digit causes them too.
- `data/labels/sudoku_corners.csv`: reviewed grid corners. Required for every training image; never train on predicted crops.
- `data/testset/`: independent test set (never train or tune on it; see its README), with reviewed `corners.csv`. `local/` holds non-redistributable images and is gitignored. Evaluate with `scripts/eval_cnn_ocr.py --labels data/testset/labels.json` and `scripts/eval_corner_cnn.py --model <onnx>`.
- `data/raw/images/`: training screenshots and photos; non-redistributable ones are in gitignored `local/` (labels keyed `local/<name>`). OCR and Corner CNN training both read these with `sudoku_corners.csv`; the ML Backend's `/fit` writes corners there. `data/yolo_grid/`: legacy YOLO keypoint data.

## Environment Variables

- `CNN_MODEL_PATH`: **required** for the web app (startup fails fast) and tests; the monitors' `--model` defaults to it.
- `CNN_BLANK_THRESHOLD` (0.65), `CNN_DIGIT_THRESHOLD` (0.55), `CNN_TOPK_CANDIDATES` (4): optional web overrides.
- `CORNER_CNN_MODEL_PATH`: web Corner CNN (defaults to the release symlink). `CORNER_CNN_MODEL`: ML Backend (required).
- `SUDOKU_DATA_DIR`: enables upload records, `/api/v1/stats` and image storage (unset locally: nothing is written). `SUDOKU_RECOGNIZE_SLOTS` (2) concurrent recognitions, `SUDOKU_RECOGNIZE_PER_MINUTE` (20) uploads per IP.
