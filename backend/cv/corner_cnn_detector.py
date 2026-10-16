"""Corner regression CNN detector: locates 4 sudoku grid corners via ONNX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .detector import DetectionResult, warp_grid
from .grid_detector import has_regular_grid_lines

LOGGER = logging.getLogger(__name__)

DEFAULT_CORNER_CNN_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "releases"
    / "sudoku_corner_cnn.onnx"
)

INPUT_SIZE = 256
REFINE_MARGIN = 0.25  # of the quad's extent, on each side


class CornerCnnDetector:
    """Detect sudoku grid corners using a lightweight regression CNN (ONNX).

    Returns 4 corner coordinates (TL, TR, BR, BL) in original image space.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
    ) -> None:
        self._session = None
        self._input_name = ""
        self._load_error: Optional[str] = None
        self.last_confidence: float = 0.0

        path = Path(model_path) if model_path else DEFAULT_CORNER_CNN_MODEL
        self._load_model(path)

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _load_model(self, path: Path) -> None:
        if not path.exists():
            self._load_error = f"Corner CNN model not found: {path}"
            return
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        LOGGER.info("Corner CNN detector loaded: %s", path.name)

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        corners = self.predict(image)
        if corners is None:
            return None
        # A plausible quadrilateral is not evidence of a Sudoku grid.
        gray = cv2.cvtColor(warp_grid(image, corners), cv2.COLOR_BGR2GRAY)
        if not has_regular_grid_lines(gray):
            return None

        return DetectionResult(
            corners=corners,
            detector_name="corner_cnn",
        )

    def predict(self, image: np.ndarray) -> np.ndarray | None:
        """Predicted corners (TL, TR, BR, BL) in image coordinates, or None if not a convex quad."""
        h_orig, w_orig = image.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(image)

        outputs = self._session.run(None, {self._input_name: blob})
        raw = np.asarray(outputs[0])[0]  # shape: (8,)

        # raw is [x_tl, y_tl, x_tr, y_tr, x_br, y_br, x_bl, y_bl] in [0, 1]
        # relative to the 256x256 letterboxed input
        corners_norm = raw.reshape(4, 2)

        # Map from letterbox space back to original image coords
        corners_raw = np.zeros((4, 2), dtype=np.float32)
        corners = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            lbx = corners_norm[i, 0] * INPUT_SIZE
            lby = corners_norm[i, 1] * INPUT_SIZE
            corners_raw[i, 0] = (lbx - pad_x) / scale
            corners_raw[i, 1] = (lby - pad_y) / scale
            corners[i, 0] = max(0.0, min(corners_raw[i, 0], w_orig - 1))
            corners[i, 1] = max(0.0, min(corners_raw[i, 1], h_orig - 1))

        if not np.isfinite(corners_raw).all() or not cv2.isContourConvex(corners):
            return None
        self.last_confidence = self._geometric_confidence(corners_raw, corners, w_orig, h_orig)
        return corners

    def refine(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
        """Corners predicted again on a crop around `corners`, or None.

        At INPUT_SIZE a grid that fills a small part of a photo spans a few
        pixels per cell, and its corners can land a cell off; in the crop the
        grid fills most of the input. Not always closer (some grids moved by
        several percent on the training images), so callers keep both.
        """
        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        pad = REFINE_MARGIN * max(x1 - x0, y1 - y0)
        h, w = image.shape[:2]
        x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
        x1, y1 = int(min(w, x1 + pad + 1)), int(min(h, y1 + pad + 1))
        refined = self.predict(image[y0:y1, x0:x1])
        return None if refined is None else refined + np.float32([x0, y0])

    @staticmethod
    def _geometric_confidence(
        corners_raw: np.ndarray, corners_clipped: np.ndarray, img_w: int, img_h: int
    ) -> float:
        """Derive a [0, 1] ranking score from geometric sanity of predicted corners.

        This is not a calibrated confidence. It is only a sorting signal for
        active learning: out-of-bounds and border-hugging quads should rank
        lower for manual review.  Callers reject non-convex quads beforehand.
        """
        width = max(float(img_w), 1.0)
        height = max(float(img_h), 1.0)

        overshoot = np.column_stack(
            [
                np.maximum(-corners_raw[:, 0], 0.0) + np.maximum(corners_raw[:, 0] - (img_w - 1), 0.0),
                np.maximum(-corners_raw[:, 1], 0.0) + np.maximum(corners_raw[:, 1] - (img_h - 1), 0.0),
            ]
        )
        overshoot_norm = overshoot / np.array([width, height], dtype=np.float32)
        in_bounds_score = max(0.0, 1.0 - float(np.mean(np.linalg.norm(overshoot_norm, axis=1))) / 0.10)

        min_border_dist = np.minimum.reduce(
            [
                corners_clipped[:, 0],
                corners_clipped[:, 1],
                (img_w - 1) - corners_clipped[:, 0],
                (img_h - 1) - corners_clipped[:, 1],
            ]
        )
        min_dim = max(min(width, height), 1.0)
        edge_margin = float(np.min(min_border_dist)) / min_dim
        edge_margin_score = float(np.clip(edge_margin / 0.05, 0.0, 1.0))

        return round(float(0.45 * in_bounds_score + 0.35 + 0.20 * edge_margin_score), 4)

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, float, float]:
        """Letterbox resize to INPUT_SIZE x INPUT_SIZE, return (blob, scale, pad_x, pad_y)."""
        h, w = image.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_x = (INPUT_SIZE - new_w) / 2.0
        pad_y = (INPUT_SIZE - new_h) / 2.0
        top = int(round(pad_y - 0.1))
        left = int(round(pad_x - 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, INPUT_SIZE - new_h - top,
            left, INPUT_SIZE - new_w - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # HWC BGR → CHW RGB, float32 [0, 1]
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], scale, float(left), float(top)
