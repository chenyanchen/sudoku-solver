# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Sudoku solver that reads a puzzle from an image. Four front ends share one recognition pipeline: a FastAPI web app (the public site, sudokusolver.siliconseries.com), a macOS screen overlay, a camera monitor prototype, and a Label Studio ML Backend. Python 3.13, `uv`. No linter or formatter is configured.

**detect grid → warp 450×450 → `backend/ocr/pipeline.recognize()` (OCR → decode → unique solution) → display**

`GLOSSARY.md` defines the domain terms (givens, reading, unique reading, misread, ambiguous vs unsolvable, release gate, test set); use them as defined there. `README.md` is partly out of date (old endpoint names, model paths and training commands); trust the code and this file over it.

## Commands

```bash
uv sync --extra dev        # pytest + ML Backend deps; other extras: ml (training), labeling, screen (overlay), viz (notebooks)
export CNN_MODEL_PATH=models/releases/sudoku_digit_cnn.onnx   # required by the app, the monitors and the tests

uv run uvicorn backend.main:app --reload        # http://localhost:8000
uv run --extra screen python -m screen_monitor --debug
CORNER_CNN_MODEL=models/releases/sudoku_corner_cnn.onnx uv run --extra labeling python -m ml_backend --port 9090

uv run pytest                                   # all tests
uv run pytest tests/test_pipeline.py -k name    # one test
```

Training/export tests (`tests/conftest.py` lists them) are skipped unless the `ml` extra is installed.

Training, export and release scripts must run from the repo root (`scripts/_tracking.py` keeps `mlruns/` paths relative to it) and refuse uncommitted changes (`--allow-dirty` records the diff instead).

### OCR model: train → gate → release

```bash
uv run --extra ml python scripts/train_cnn_ocr.py --version <MAJOR.MINOR.PATCH> --epochs 60 --patience 20 --corner-jitter-copies 3 --seed <n>
uv run --extra ml python scripts/export_cnn_onnx.py --checkpoint <run .pt> --output <scratch>.onnx
CNN_MODEL_PATH=<candidate.onnx> uv run python scripts/eval_cnn_ocr.py --cnn-model <candidate.onnx> --strict                              # release gate: every labeled image
CNN_MODEL_PATH=<candidate.onnx> uv run python scripts/eval_cnn_ocr.py --cnn-model <candidate.onnx> --labels data/testset/labels.json   # report only
uv run --extra ml python scripts/release_model.py cnn --run-id <id> --version <YYYYMMDD.N>
```

- **Release gate**: every labeled image read exactly after decode. `release_model.py` runs it and refuses a candidate that fails or whose labeled image is missing. Release only if the candidate also has raw (pre-decode) errors no worse than the current release and `pytest` passes. The validation score saturates at 1.0 and is not a release signal.
- **The test set is reported, never gated on or tuned to.**
- Seeds alone swing results (often only one of three seeds passes the gate), so compare architectures or data over 3 seeds each, drawn at random per comparison (`python -c 'import random; print(random.sample(range(10**6), 3))'`), not 0/1/2.
- `--corner-jitter-copies` adds warps from corners moved ≤1.5% of the grid side, because detected corners are a few pixels off. To measure robustness, warp from reviewed corners moved ±1.5% and ±3% and count decode-exact grids.
- Training uses MPS when available (separable convs are ~50× slower on CPU); the device changes results, so compare runs on one device. Output prints only at exit when redirected; MLflow's `metrics/score` shows progress.
- Versions: training `MAJOR.MINOR.PATCH` (MAJOR = architecture, MINOR = data/hyperparameters); release `YYYYMMDD.N`. `models/releases/<model>.onnx` is a symlink to the current release; each release has a `.meta.json` sidecar (version, class order, commit, seed, data hashes, val split, gate results) that stands alone without `mlruns/`.

### 16×16 OCR model

A separate 17-class model (blank, 1–9, A–G as 10–16), so the 9×9 model is untouched. Same flow with these differences:

```bash
uv run --extra ml python scripts/synth_16x16_grids.py --count 800 --fonts data/fonts --out data/synth_16x16   # gitignored; data/fonts holds Roboto
uv run --extra ml python scripts/train_cnn_ocr.py --version <v> --num-classes 17 --extra data/synth_16x16/labels.json data/synth_16x16/corners.csv --epochs 30 --patience 10 --corner-jitter-copies 1
CNN16_MODEL_PATH=<candidate.onnx> uv run python scripts/eval_cnn_ocr.py --labels data/labels/sudoku16_labels.json --cnn-model <candidate.onnx> --strict
uv run --extra ml python scripts/release_model.py cnn16 --run-id <id> --version <YYYYMMDD.N>
```

Letters are learned from synthetic grids rendered in system fonts (macOS paths) plus Roboto, through the same warp → binarize → cut path. Judge candidates on held-out *fonts*: ten families (Georgia, Courier New, Impact, Skia, Comic Sans…) stay out of `SYSTEM_FONTS` for that. Synthetic held-out puzzles must have unique solutions, or the 16×16 reading never wins over the 9×9 one.

### Corner CNN: train → eval → release

```bash
uv run --extra ml python scripts/synth_corner_data.py --count 6000 --seed 0          # → gitignored data/synth_corner*/
uv run --extra ml python scripts/train_corner_cnn.py --synth data/synth_corner --version <MAJOR.MINOR.PATCH>
uv run --extra ml python scripts/export_corner_cnn_onnx.py --checkpoint <run .pt> --output <scratch>.onnx
uv run python scripts/eval_corner_cnn.py --model <candidate.onnx>      # held-out test set; compare with the release
uv run --extra ml python scripts/release_model.py corner-cnn --run-id <id> --version <YYYYMMDD.N>
```

- The val split is a hash of each puzzle, so adding images keeps existing ones on their side. To judge new data, train a control on the old corners CSV (`--corners`) and compare on the shared val images.
- Pick between candidates on training/val error (`eval_corner_cnn.py --labels data/labels/sudoku_labels.json --corners data/labels/sudoku_corners.csv`), not the test set. Synthetic images are training-only.
- Also run `eval_cnn_ocr.py --strict` with `CORNER_CNN_MODEL_PATH=<candidate>`: closer corners have still produced warps the OCR misread into a wrong unique solution.
- Through `read_image` most models read the same images (OpenCV reads ~90%), so end-to-end counts are too few to rank Corner CNN candidates; use corner error.
- The current architecture is `tiny_32-64-128-256_ctx` at 384px (heatmap + soft-argmax). Regressing corners from pooled features underfit; small nets without the stride-32 context branch or keypad synth data occasionally put a corner on a keypad beside the grid.

### Deploy

```bash
deploy/deploy.sh <host>          # ships the committed HEAD (git archive), keeps 3 releases, restarts, rescores uploads
```

The pages are on GitHub Pages behind Cloudflare; `.github/workflows/pages.yml` runs `scripts/build_static_site.py` on every push to `main` that touches `frontend/`, `backend/i18n.py` or the build script. The server (one 2-core/2 GB VPS: uvicorn with 1 worker behind Caddy, systemd `sudoku-solver`) is only the API. First-time setup is `deploy/setup-server.sh [domain]`.

After each deploy, `python -m backend.rescore --version <rev>` reads every kept upload again into the `rescores` table, so the public solve rate is what the deployed release solves. Whether a release generalizes is judged on the test set, not on this rate: most uploads end up in training. No access log is kept: the privacy note promises no IPs are stored.

## Architecture

### Reading an image (`backend/api/routes.py: read_image`)

The unique solution is the acceptance test throughout: a reading is trusted only when its givens have exactly one solution, and candidates are tried in order until one does.

1. **OpenCV contours** (`backend/cv/grid_detector.py`, wrapped by `opencv_detector.py`). Fixed-pixel thresholds suit no single scale, so it tries full resolution, then a 1280px-wide copy, and takes the first warp with regular grid lines (`has_regular_grid_lines`). A quad without regular lines is usually a decorative frame or a single 3×3 box; `snap_to_lattice` fits the 9×9 line lattice around it (also for borderless grids and grids cropped by the image edge). Fitting costs ~100× checking a quad, so quads are snapped only after no variant has a regular one.
2. **Corner CNN** (`backend/cv/corner_cnn_detector.py`), when OpenCV's grid is not a unique reading. Each solves images the other cannot. It skips the regularity check (which rejects good corners on photos with moiré); a quad that does not solve is snapped to the lattice. `refine` predicts again on a crop around the first quad; the refined reading is always tried and wins with the same solution and more givens.
3. **16×16** (`_read_16x16`), only when no 9×9 reading is unique, on the quads the 9×9 detectors found. It replaces the 9×9 reading only if it solves uniquely or no 9×9 grid was found, so a solved 9×9 grid never changes. Without the 16×16 model only 9×9 grids are read.
4. **Rotation**: a reading that cannot be the puzzle as printed is retried on the image turned 90° either way; a turned reading is kept only if unique.

When nothing is unique, `_rank` picks what to show: an ambiguous reading with ≥17 givens (some published puzzles are ambiguous) beats a warp with regular lines, which beats one without (more givens win there). OCR always re-warps from the full-resolution frame.

### Pipeline (`backend/ocr/pipeline.py`)

The only OCR → solve path, used by every front end and by training and eval:

- `prepare_grid_for_ocr`: adaptive-threshold binarization. Training uses the same function; per-source gray/CLAHE variants measured far worse.
- `ocr_cells`: cuts cells with a 1px margin; zooms in when the grid's median digit height is below the training range (small-font screenshots) and lowers the digit threshold for that grid.
- `decode`: if the givens are not a unique reading, searches ≤2 edits over the CNN's candidates among the 10 most doubtful cells (fill missed givens if ambiguous, replace misreads if unsolvable); a correction is trusted only if no other with as few edits gives a different solution. Cells read as blank with blank probability < 0.05 start as givens with their top candidate, outside that budget.

`backend/ocr/cnn_digit_reader.py` runs the CNN on 48×48 cells with ONNX Runtime and raises at construction if the model can't load (the web app fails fast at startup). Architectures are in `backend/ocr/model_arch.py` and `backend/cv/corner_cnn_arch.py`, shared by training and export.

### Solver (`backend/solver/backtracking.py`)

Bitmask backtracking with MRV and a 100K-node budget, for 9×9 and 16×16. `solve_unique` returns a solution only if it is unique (an exhausted budget counts as a second solution). `/api/v1/sudoku:solve` answers an ambiguous grid with status `ambiguous`, one full solution and the `undetermined` cells (those that differ between solutions); the page hatches those cells and never blanks them.

### Web

- API (`routes.py`): `sudoku:recognize` (image → reading; rate-limited per IP, bounded concurrency), `sudoku:solve`, `sudoku:feedback` (the uploader's correction), `stats`. Recognition and solving are separate calls; the page edits the reading in between.
- `backend/records.py`: with `SUDOKU_DATA_DIR` set, every upload's reading, corrections and rescores go to `records.sqlite3`, and each image is kept once per SHA-256. Stats count distinct images by their latest outcome. Unset locally: nothing is written.
- Pages (`backend/i18n.py`): `/` and `/zh/` render `frontend/index.html`; articles are `frontend/article.html` around `frontend/articles/<slug>.<lang>.html`. The same renderer serves the pages locally and builds the static site, where API calls go to `i18n.API_URL` (`window.SITE`). With `SUDOKU_REDIRECT_PAGES` (set on the server) page URLs 301 to the public site. `frontend/sw.js` receives the PWA share target and hands the image to the page through the Cache API.
- Hostnames are one level under siliconseries.com: Cloudflare's free certificate covers only `*.siliconseries.com`.

### Other front ends

- **Screen monitor** (`screen_monitor/agent.py`): capture loop in a daemon thread, Qt overlay on the main thread. OpenCV detection only, at ≤1280px. Detects/OCRs only when the frame changed or a rescan is due; OCR results are cached by the warped grid's visual hash; hints appear after N stable frames. OpenTelemetry → Prometheus, dashboards in `grafana/`.
- **ML Backend** (`ml_backend/`): Label Studio protocol (`/health`, `/setup`, `/predict`, `/fit`) for corner keypoint and OCR projects. Requires `CNN_MODEL_PATH` and `CORNER_CNN_MODEL` with `.meta.json` sidecars; reports model version `<cnn>+<corner>`. `/fit` writes reviewed corners to `data/labels/sudoku_corners.csv`. Label Studio bug [#7151](https://github.com/HumanSignal/label-studio/issues/7151): the "Prediction score" column needs `project.model_version` patched to that exact value (`PATCH /api/projects/<id>` with `{"model_version": "<version from /setup>"}`).
- **Camera monitor** (`camera_monitor/`): prototype. Its YOLO model was removed in 90dcddd, so it does not run as shipped; the YOLO scripts and `backend/cv/yolo_detector.py` are legacy.

## Data

- `data/labels/sudoku_labels.json`: the givens as printed, for every training image (also the release gate). Every label must have at least one solution: zero means a transcription error. Several solutions are allowed when the puzzle itself is ambiguous; check such a label against the image, since a missed digit causes them too.
- `data/labels/sudoku_corners.csv`: reviewed grid corners (TL, TR, BR, BL, normalized; may lie past the image edge). Required for every training image; never train on predicted corners.
- `data/raw/images/`: training images; non-redistributable ones are in gitignored `local/` (labels keyed `local/<name>`).
- `data/testset/`: independent test set with its own `labels.json` and reviewed `corners.csv`. Never train or tune on it, never add its images to the training labels (`tests/test_testset.py` enforces this), and report the score before and after any change made because of a test-set failure. Its README logs how uploads are split by puzzle between it and training.
- `data/labels/sudoku16_labels.json` / `sudoku16_corners.csv`: real 16×16 uploads, the 16×16 model's gate.
- `data/production/` (gitignored): a local copy of the server's `SUDOKU_DATA_DIR` for looking at real uploads. Refresh from a consistent snapshot:
  ```bash
  ssh <host> 'sqlite3 /var/lib/sudoku-solver/records.sqlite3 ".backup /tmp/records-snapshot.sqlite3"' && rsync -a <host>:/tmp/records-snapshot.sqlite3 data/production/records.sqlite3 && rsync -a <host>:/var/lib/sudoku-solver/images/ data/production/images/ && ssh <host> 'rm /tmp/records-snapshot.sqlite3'
  ```
- Gitignored and regenerated by scripts: `data/synth_corner*/`, `data/synth_16x16/`, `data/fonts/`, `mlruns/`, `models/checkpoints/`.

## Environment variables

- `CNN_MODEL_PATH`: **required** for the web app and tests; the monitors' `--model` defaults to it.
- `CNN16_MODEL_PATH`, `CORNER_CNN_MODEL_PATH`: web 16×16 OCR and Corner CNN models; both default to the release symlinks. `CORNER_CNN_MODEL`: ML Backend (required there).
- `CNN_BLANK_THRESHOLD` (0.65), `CNN_DIGIT_THRESHOLD` (0.55), `CNN_TOPK_CANDIDATES` (4): optional web overrides.
- `SUDOKU_DATA_DIR`: enables upload records, stats and image storage. `SUDOKU_RECOGNIZE_SLOTS` (2) concurrent recognitions, `SUDOKU_RECOGNIZE_PER_MINUTE` (20) uploads per IP.
- `SUDOKU_BASE_URL` (public site), `SUDOKU_API_URL` (server), `SUDOKU_REDIRECT_PAGES` (server pages redirect to the public site).
