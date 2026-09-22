"""Corner regression CNN detector: locates 4 sudoku grid corners via ONNX."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .detector import DetectionResult

LOGGER = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_CORNER_CNN_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "releases"
    / "sudoku_corner_cnn.onnx"
)

INPUT_SIZE = 256


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
        if ort is None:
            self._load_error = "onnxruntime not installed"
            return
        if not path.exists():
            self._load_error = f"Corner CNN model not found: {path}"
            return
        try:
            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            LOGGER.info("Corner CNN detector loaded: %s", path.name)
        except Exception as exc:
            self._load_error = str(exc)

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        if not self.is_ready:
            return None

        h_orig, w_orig = image.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(image)

        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0][0]  # shape: (8,)

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

        self.last_confidence = self._geometric_confidence(corners_raw, corners, w_orig, h_orig)

        return DetectionResult(
            corners=corners,
            detector_name="corner_cnn",
        )

    @staticmethod
    def _geometric_confidence(
        corners_raw: np.ndarray, corners_clipped: np.ndarray, img_w: int, img_h: int
    ) -> float:
        """Derive a [0, 1] ranking score from geometric sanity of predicted corners.

        This is not a calibrated confidence. It is only a sorting signal for
        active learning: out-of-bounds, non-convex, and border-hugging quads
        should rank lower for manual review.
        """
        n = len(corners_clipped)
        cross_signs = []
        for i in range(n):
            p0 = corners_clipped[i]
            p1 = corners_clipped[(i + 1) % n]
            p2 = corners_clipped[(i + 2) % n]
            a = p1 - p0
            b = p2 - p1
            cross_signs.append(float(a[0] * b[1] - a[1] * b[0]))
        is_convex = all(c > 0 for c in cross_signs) or all(c < 0 for c in cross_signs)
        if not is_convex:
            return 0.0

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

        return round(float(0.45 * in_bounds_score + 0.35 * 1.0 + 0.20 * edge_margin_score), 4)

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
