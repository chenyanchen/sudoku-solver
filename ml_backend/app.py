"""FastAPI ML Backend implementing Label Studio's ML Backend protocol.

Serves both project types:
  - Keypoints: YOLO grid corner detection
  - OCR: YOLO warp → CNN digit recognition → solve verification

Endpoints:
  GET  /health  - health check
  POST /setup   - called on project connection
  POST /predict - return predictions for tasks
  POST /fit     - save submitted annotations back to repo
"""

from __future__ import annotations

import base64
from contextlib import asynccontextmanager
import json
import logging
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI

REPO_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger("ml_backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Release configuration
# ---------------------------------------------------------------------------
_yolo = None
_cnn = None

_CNN_ENV = "CNN_MODEL_PATH"
_YOLO_ENV = "YOLO_MODEL"


def _validate_backend_model_env() -> tuple[Path, Path]:
    cnn_path = os.getenv(_CNN_ENV)
    yolo_path = os.getenv(_YOLO_ENV)

    missing = [name for name, value in ((_CNN_ENV, cnn_path), (_YOLO_ENV, yolo_path)) if not value]
    if missing:
        raise RuntimeError(
            "ML Backend requires explicit release model paths: "
            + ", ".join(missing)
        )

    cnn = Path(cnn_path)
    yolo = Path(yolo_path)
    for path, env_name in ((cnn, _CNN_ENV), (yolo, _YOLO_ENV)):
        if not path.exists():
            raise RuntimeError(f"{env_name} points to a missing file: {path}")
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            raise RuntimeError(f"Release metadata file is missing for {env_name}: {meta_path}")

    return cnn, yolo


def _read_release_metadata(model_path: Path) -> dict[str, Any]:
    meta_path = model_path.with_suffix(".meta.json")
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Release metadata file is missing: {meta_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Release metadata file is invalid JSON: {meta_path}") from exc

    version = payload.get("model_version") or payload.get("release_version") or payload.get("version")
    if not isinstance(version, str) or not version.strip():
        raise RuntimeError(f"Release metadata file has no model version: {meta_path}")
    payload["model_version"] = version.strip()
    return payload


def _current_release_versions() -> tuple[str, str]:
    cnn_path, yolo_path = _validate_backend_model_env()
    cnn_version = _read_release_metadata(cnn_path)["model_version"]
    yolo_version = _read_release_metadata(yolo_path)["model_version"]
    return cnn_version, yolo_version


def _response_model_version(label_config: str) -> str:
    cnn_version, yolo_version = _current_release_versions()
    if _is_keypoints_project(label_config):
        return yolo_version
    return f"{cnn_version}+{yolo_version}"


def _get_yolo():
    global _yolo
    if _yolo is None:
        _, yolo_path = _validate_backend_model_env()
        from backend.cv.yolo_detector import YoloGridDetector
        _yolo = YoloGridDetector(
            model_path=yolo_path,
            num_keypoints=16,
        )
        if _yolo.is_ready:
            logger.info("YOLO loaded (16-kpt)")
        else:
            logger.error("YOLO failed: %s", _yolo.load_error)
    return _yolo


def _get_cnn():
    global _cnn
    if _cnn is None:
        cnn_path, _ = _validate_backend_model_env()
        from backend.ocr.cnn_digit_reader import CnnDigitReader
        _cnn = CnnDigitReader(model_path=cnn_path, strict=False)
        if _cnn.is_ready:
            logger.info("CNN loaded: %s", _cnn.model_version)
        else:
            logger.error("CNN failed: %s", _cnn.load_error)
    return _cnn


@asynccontextmanager
async def _app_lifespan(_: FastAPI):
    _validate_backend_model_env()
    yolo = _get_yolo()
    cnn = _get_cnn()
    if not yolo.is_ready:
        raise RuntimeError(f"Failed to load YOLO model at startup: {yolo.load_error}")
    if not cnn.is_ready:
        raise RuntimeError(f"Failed to load CNN model at startup: {cnn.load_error}")
    yield


app = FastAPI(title="Sudoku ML Backend", lifespan=_app_lifespan)


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def _decode_image(image_ref: str, ls_url: str | None = None, ls_token: str | None = None) -> np.ndarray | None:
    """Decode image from data URI or Label Studio URL."""
    if image_ref.startswith("data:"):
        # data:image/jpeg;base64,/9j/...
        _, encoded = image_ref.split(",", 1)
        buf = np.frombuffer(base64.b64decode(encoded), np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if image_ref.startswith(("/data/", "http")):
        url = image_ref if image_ref.startswith("http") else f"{ls_url}{image_ref}"
        req = urllib.request.Request(url)
        if ls_token:
            prefix = "Bearer" if ls_token.startswith("eyJ") else "Token"
            req.add_header("Authorization", f"{prefix} {ls_token}")
        with urllib.request.urlopen(req) as resp:
            buf = np.frombuffer(resp.read(), np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Local file path.
    return cv2.imread(str(image_ref))


def _extract_filename(task: dict) -> str:
    """Best-effort extraction of original filename from task data."""
    image_ref = task.get("data", {}).get("image", "")
    basename = image_ref.rstrip("/").split("/")[-1]
    # Strip Label Studio upload hash prefix: "abc123-filename.jpg" → "filename.jpg"
    match = re.match(r"^[a-f0-9]+-(.+)$", basename)
    if match:
        return match.group(1)
    # Strip _warped suffix for OCR project.
    stem = Path(basename).stem.removesuffix("_warped")
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = REPO_ROOT / "data" / "raw" / "images" / f"{stem}{ext}"
        if candidate.exists():
            return f"{stem}{ext}"
    return basename


# ---------------------------------------------------------------------------
# Project type detection
# ---------------------------------------------------------------------------

def _is_keypoints_project(label_config: str) -> bool:
    return "KeyPointLabels" in label_config


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

# 16-keypoint labels: 4x4 grid, row-major.  P{row}{col}.
_KPT_LABELS_16 = [f"P{r}{c}" for r in range(4) for c in range(4)]


def _predict_keypoints(image: np.ndarray) -> tuple[list[dict], float]:
    yolo = _get_yolo()
    if not yolo.is_ready:
        return [], 0.0
    det = yolo.detect(image)
    if det is None:
        return [], 0.0

    if det.keypoints is None or det.keypoints.shape != (16, 2):
        logger.warning("predict/keypoints: detector returned invalid keypoints shape: %s", None if det.keypoints is None else det.keypoints.shape)
        return [], 0.0

    conf = yolo.last_confidence
    h, w = image.shape[:2]
    keypoints = det.keypoints

    result = []
    for i, label in enumerate(_KPT_LABELS_16):
        result.append({
            "from_name": "corners",
            "to_name": "image",
            "type": "keypointlabels",
            "value": {
                "x": round(float(keypoints[i][0]) / w * 100, 4),
                "y": round(float(keypoints[i][1]) / h * 100, 4),
                "width": 0.3,
                "keypointlabels": [label],
            },
            "score": round(float(conf), 4),
        })
    return result, float(conf)


def _predict_ocr(image: np.ndarray) -> tuple[list[dict], float]:
    yolo = _get_yolo()
    cnn = _get_cnn()
    if not cnn.is_ready:
        return [], 0.0

    # Try YOLO warp first (raw photo). If it fails, treat as already warped.
    if yolo.is_ready:
        det = yolo.detect(image)
    else:
        det = None

    if det is not None:
        from backend.cv.detector import warp_grid
        warped = warp_grid(image, det.corners)
    else:
        warped = cv2.resize(image, (450, 450))

    from screen_monitor.solver_pipeline import ocr_grid_from_warped
    ocr_result, reason, givens = ocr_grid_from_warped(warped, cnn, min_givens=0)
    if ocr_result is None:
        return [], 0.0

    grid = ocr_result["original_grid"]
    avg_conf = ocr_result.get("metadata", {}).get("average_confidence", 0.0) or 0.0

    from backend.solver.backtracking import SudokuSolver
    solved = SudokuSolver().solve(grid) is not None
    score = avg_conf if solved else 0.0

    result = [{
        "from_name": "grid",
        "to_name": "image",
        "type": "textarea",
        "value": {"text": [" ".join(str(d) for d in row) for row in grid]},
        "score": round(score, 4),
    }]
    return result, round(score, 4)


# ---------------------------------------------------------------------------
# /fit
# ---------------------------------------------------------------------------

def _fit_keypoints(annotation: dict, task: dict) -> None:
    """Save corrected 16 keypoints to YOLO pose label format."""
    from scripts.auto_label_yolo import keypoints_16_to_yolo_pose

    filename = _extract_filename(task)
    src_image = REPO_ROOT / "data" / "raw" / "images" / filename
    if not src_image.exists():
        logger.warning("fit/keypoints: source image not found: %s", filename)
        return

    image = cv2.imread(str(src_image))
    if image is None:
        return
    h, w = image.shape[:2]

    # Parse 16 keypoints from annotation (labels: P00..P33).
    kpt_map: dict[str, tuple[float, float]] = {}
    seen_labels: list[str] = []
    for r in annotation.get("result", []):
        if r.get("type") != "keypointlabels":
            continue
        labels = r["value"].get("keypointlabels", [])
        if not labels:
            continue
        x_abs = r["value"]["x"] / 100.0 * w
        y_abs = r["value"]["y"] / 100.0 * h
        label = labels[0]
        seen_labels.append(label)
        kpt_map[label] = (x_abs, y_abs)

    missing = [label for label in _KPT_LABELS_16 if label not in kpt_map]
    duplicates = sorted({label for label in seen_labels if seen_labels.count(label) > 1})
    if missing or duplicates:
        logger.warning(
            "fit/keypoints: invalid keypoint labels for %s missing=%s duplicates=%s",
            filename,
            missing,
            duplicates,
        )
        return

    keypoints = np.array([kpt_map[label] for label in _KPT_LABELS_16], dtype=np.float32)

    label_line = keypoints_16_to_yolo_pose(keypoints, w, h)

    lbl_dir = REPO_ROOT / "data" / "yolo_grid" / "labels" / "train"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / f"{Path(filename).stem}.txt").write_text(label_line + "\n")
    logger.info("fit/keypoints: saved %s (16-kpt)", filename)


def _fit_ocr(annotation: dict, task: dict) -> None:
    """Save corrected OCR grid to sudoku_labels.json."""
    filename = _extract_filename(task)

    # Parse TextArea result.
    grid_lines: list[str] = []
    for r in annotation.get("result", []):
        if r.get("from_name") == "grid" and r.get("type") == "textarea":
            grid_lines = r["value"].get("text", [])
            break

    if not grid_lines:
        logger.warning("fit/ocr: no grid text for %s", filename)
        return

    grid: list[list[int]] = []
    for line in grid_lines:
        row = [int(d) for d in line.strip().split()]
        if len(row) != 9:
            logger.warning("fit/ocr: bad row length %d for %s", len(row), filename)
            return
        grid.append(row)
    if len(grid) != 9:
        logger.warning("fit/ocr: bad grid rows %d for %s", len(grid), filename)
        return

    labels_path = REPO_ROOT / "data" / "labels" / "sudoku_labels.json"
    labels: dict = {}
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())

    labels[filename] = grid
    labels_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    logger.info("fit/ocr: saved %s (%d givens)", filename,
                sum(1 for row in grid for c in row if c != 0))


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "UP", "model_class": "SudokuMLBackend"}


@app.post("/setup")
def setup(payload: dict[str, Any] = {}) -> dict:
    # Warm up models.
    _get_yolo()
    _get_cnn()
    return {"model_version": _response_model_version(payload.get("label_config", ""))}


@app.post("/predict")
def predict(payload: dict[str, Any] = {}) -> dict:
    tasks = payload.get("tasks", [])
    label_config = payload.get("label_config", "")
    is_keypoints = _is_keypoints_project(label_config)
    model_version = _response_model_version(label_config)

    ls_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
    ls_token = os.getenv("LABEL_STUDIO_API_KEY", "")
    # Resolve JWT refresh token if needed.
    if ls_token.startswith("eyJ"):
        try:
            body = json.dumps({"refresh": ls_token}).encode()
            req = urllib.request.Request(f"{ls_url}/api/token/refresh", data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req) as resp:
                ls_token = json.loads(resp.read())["access"]
        except Exception:
            pass

    results = []
    for task in tasks:
        image_ref = task.get("data", {}).get("image", "")
        image = _decode_image(image_ref, ls_url, ls_token)
        if image is None:
            results.append({"result": [], "score": 0.0, "model_version": model_version})
            continue

        if is_keypoints:
            result, score = _predict_keypoints(image)
        else:
            result, score = _predict_ocr(image)

        results.append({
            "result": result,
            "score": score,
            "model_version": model_version,
        })

    return {"results": results}


@app.post("/fit")
def fit(payload: dict[str, Any] = {}) -> dict:
    annotation = payload.get("annotation", {})
    task = payload.get("task", annotation.get("task", {}))
    if isinstance(task, int):
        # Only got task ID, not full task data — skip.
        logger.warning("fit: received task ID only, cannot extract filename")
        return {"status": "skipped"}

    label_config = payload.get("label_config", "")
    if not label_config:
        # Try to infer from annotation content.
        has_keypoints = any(
            r.get("type") == "keypointlabels" for r in annotation.get("result", [])
        )
        label_config = "KeyPointLabels" if has_keypoints else ""

    if _is_keypoints_project(label_config):
        _fit_keypoints(annotation, task)
    else:
        _fit_ocr(annotation, task)

    return {"status": "ok"}
