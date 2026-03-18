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

app = FastAPI(title="Sudoku ML Backend")

# ---------------------------------------------------------------------------
# Model singletons (lazy-loaded)
# ---------------------------------------------------------------------------
_yolo = None
_cnn = None


def _get_yolo():
    global _yolo
    if _yolo is None:
        from backend.cv.yolo_detector import YoloGridDetector
        _yolo = YoloGridDetector(model_path=os.getenv("YOLO_MODEL"))
        if _yolo.is_ready:
            logger.info("YOLO loaded")
        else:
            logger.error("YOLO failed: %s", _yolo.load_error)
    return _yolo


def _get_cnn():
    global _cnn
    if _cnn is None:
        from backend.ocr.cnn_digit_reader import CnnDigitReader
        _cnn = CnnDigitReader(strict=False)
        if _cnn.is_ready:
            logger.info("CNN loaded: %s", _cnn.model_version)
        else:
            logger.error("CNN failed: %s", _cnn.load_error)
    return _cnn


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

def _predict_keypoints(image: np.ndarray) -> tuple[list[dict], float]:
    yolo = _get_yolo()
    if not yolo.is_ready:
        return [], 0.0
    det = yolo.detect(image)
    if det is None:
        return [], 0.0

    _, corners, conf = det
    h, w = image.shape[:2]

    result = []
    for i, label in enumerate(["TL", "TR", "BR", "BL"]):
        result.append({
            "from_name": "corners",
            "to_name": "image",
            "type": "keypointlabels",
            "value": {
                "x": round(float(corners[i][0]) / w * 100, 4),
                "y": round(float(corners[i][1]) / h * 100, 4),
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
        warped, _, _ = det
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
    """Save corrected keypoints to YOLO pose label format."""
    from scripts.auto_label_yolo import corners_to_yolo_pose

    filename = _extract_filename(task)
    src_image = REPO_ROOT / "data" / "raw" / "images" / filename
    if not src_image.exists():
        logger.warning("fit/keypoints: source image not found: %s", filename)
        return

    image = cv2.imread(str(src_image))
    if image is None:
        return
    h, w = image.shape[:2]

    # Parse keypoints from annotation.
    corner_map: dict[str, tuple[float, float]] = {}
    for r in annotation.get("result", []):
        if r.get("type") != "keypointlabels":
            continue
        labels = r["value"].get("keypointlabels", [])
        if not labels:
            continue
        x_abs = r["value"]["x"] / 100.0 * w
        y_abs = r["value"]["y"] / 100.0 * h
        corner_map[labels[0]] = (x_abs, y_abs)

    if len(corner_map) < 4:
        logger.warning("fit/keypoints: only %d corners for %s", len(corner_map), filename)
        return

    corners = np.float32([corner_map[l] for l in ["TL", "TR", "BR", "BL"]])
    label_line = corners_to_yolo_pose(corners, w, h)

    lbl_dir = REPO_ROOT / "data" / "yolo_grid" / "labels" / "train"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    (lbl_dir / f"{Path(filename).stem}.txt").write_text(label_line + "\n")
    logger.info("fit/keypoints: saved %s", filename)


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
    return {"model_version": "sudoku-v1"}


@app.post("/predict")
def predict(payload: dict[str, Any] = {}) -> dict:
    tasks = payload.get("tasks", [])
    label_config = payload.get("label_config", "")
    is_keypoints = _is_keypoints_project(label_config)

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
            results.append({"result": [], "score": 0.0, "model_version": "sudoku-v1"})
            continue

        if is_keypoints:
            result, score = _predict_keypoints(image)
        else:
            result, score = _predict_ocr(image)

        results.append({
            "result": result,
            "score": score,
            "model_version": "sudoku-v1",
        })

        # Write score to task data so Data Manager can sort by it.
        # (Label Studio's native Prediction Score is broken in v1.20.)
        task_id = task.get("id")
        if task_id and ls_token:
            try:
                body = json.dumps({"data": {**task.get("data", {}), "score": score}}).encode()
                req = urllib.request.Request(
                    f"{ls_url}/api/tasks/{task_id}", data=body, method="PATCH",
                )
                req.add_header("Authorization", f"Bearer {ls_token}")
                req.add_header("Content-Type", "application/json")
                urllib.request.urlopen(req)
            except Exception as e:
                logger.debug("Failed to update task %s score: %s", task_id, e)

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
