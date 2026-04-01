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
        corners = np.zeros((4, 2), dtype=np.float32)
        for i in range(4):
            lbx = corners_norm[i, 0] * INPUT_SIZE
            lby = corners_norm[i, 1] * INPUT_SIZE
            corners[i, 0] = max(0.0, min((lbx - pad_x) / scale, w_orig - 1))
            corners[i, 1] = max(0.0, min((lby - pad_y) / scale, h_orig - 1))

        self.last_confidence = 1.0  # regression model has no confidence score

        return DetectionResult(
            corners=corners,
            detector_name="corner_cnn",
        )

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
