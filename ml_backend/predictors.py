from __future__ import annotations

import base64
import urllib.request
from typing import Any

import cv2
import numpy as np

from .schema import (
    extract_image_ref,
    extract_keypoint_binding,
    extract_textarea_binding,
    ordered_corner_labels,
)


def decode_image(
    image_ref: str, ls_url: str | None = None, ls_token: str | None = None
) -> np.ndarray | None:
    image_ref = (image_ref or "").strip()
    if not image_ref:
        return None

    if image_ref.startswith("data:"):
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

    return cv2.imread(str(image_ref))


def load_task_image(
    task: dict[str, Any],
    parsed_label_config: dict[str, Any],
    ls_url: str,
    ls_token: str,
) -> tuple[str, np.ndarray | None]:
    image_ref = extract_image_ref(task, parsed_label_config)
    return image_ref, decode_image(image_ref, ls_url, ls_token)


def predict_corners(
    image: np.ndarray,
    parsed_label_config: dict[str, Any],
    detector,
) -> tuple[list[dict[str, Any]], float]:
    if not detector.is_ready:
        return [], 0.0

    det = detector.detect(image)
    if det is None:
        return [], 0.0

    from_name, to_name, _labels = extract_keypoint_binding(parsed_label_config)
    labels = ordered_corner_labels(parsed_label_config)
    h, w = image.shape[:2]
    results = []
    for idx, label in enumerate(labels):
        results.append(
            {
                "from_name": from_name,
                "to_name": to_name,
                "type": "keypointlabels",
                "value": {
                    "x": round(float(det.corners[idx, 0]) / w * 100, 4),
                    "y": round(float(det.corners[idx, 1]) / h * 100, 4),
                    "width": 0.3,
                    "keypointlabels": [label],
                },
                "score": round(detector.last_confidence, 4),
            }
        )
    return results, float(detector.last_confidence)


def predict_ocr(
    image: np.ndarray,
    parsed_label_config: dict[str, Any],
    corner_cnn,
    cnn,
) -> tuple[list[dict[str, Any]], float]:
    if not cnn.is_ready:
        return [], 0.0

    det = corner_cnn.detect(image) if corner_cnn.is_ready else None
    if det is not None:
        from backend.cv.detector import warp_grid

        warped = warp_grid(image, det.corners)
    else:
        warped = cv2.resize(image, (450, 450))

    from screen_monitor.solver_pipeline import ocr_grid_from_warped
    from backend.solver.backtracking import SudokuSolver

    ocr_result, _reason, _givens = ocr_grid_from_warped(warped, cnn, min_givens=0)
    if ocr_result is None:
        return [], 0.0

    grid = ocr_result["original_grid"]
    avg_conf = ocr_result.get("metadata", {}).get("average_confidence", 0.0) or 0.0
    solved = SudokuSolver().solve(grid) is not None
    score = float(avg_conf) if solved else 0.0
    from_name, to_name = extract_textarea_binding(parsed_label_config)
    result = [
        {
            "from_name": from_name,
            "to_name": to_name,
            "type": "textarea",
            "value": {"text": [" ".join(str(d) for d in row) for row in grid]},
            "score": round(score, 4),
        }
    ]
    return result, round(score, 4)
