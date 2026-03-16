"""YOLO26n-pose grid detector: detects sudoku grid + 4 corner keypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_YOLO_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "releases"
    / "yolo_grid_detect.onnx"
)


class YoloGridDetector:
    """Detect sudoku grids using YOLO26n-pose ONNX model.

    Returns the 4 corner keypoints (TL, TR, BR, BL) in the original
    image coordinate space, ready for perspective warp.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        imgsz: int = 640,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._imgsz = imgsz
        self._session = None
        self._input_name = ""
        self._output_names: list[str] = []
        self._load_error: Optional[str] = None

        path = Path(model_path) if model_path else DEFAULT_YOLO_MODEL
        self._load_model(path)

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def detect(
        self, image: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        """Detect the best sudoku grid in an image.

        Args:
            image: BGR image (any size).

        Returns:
            (warped_450x450, corners_4x2, confidence) or None if no grid found.
        """
        if not self.is_ready:
            return None

        h_orig, w_orig = image.shape[:2]

        # Preprocess: letterbox resize to imgsz x imgsz.
        img_input, scale, pad_x, pad_y = self._preprocess(image)

        # Run inference.
        outputs = self._session.run(
            self._output_names,
            {self._input_name: img_input},
        )
        # YOLO pose output shape: (1, num_preds, 5 + num_classes + num_kpts*3)
        # For 1 class + 4 keypoints*2: (1, N, 14)
        # or (1, N, 5+1+4*3) = (1, N, 18) if visibility is included.
        preds = outputs[0]  # (1, N, C) or (1, C, N) depending on export

        # Parse detections.
        best = self._parse_best_detection(preds, scale, pad_x, pad_y, w_orig, h_orig)
        if best is None:
            return None

        corners, conf = best

        # Warp to 450x450.
        dst = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, matrix, (450, 450))

        return warped, corners, conf

    def _load_model(self, path: Path) -> None:
        if ort is None:
            self._load_error = "onnxruntime not installed"
            return
        if not path.exists():
            self._load_error = f"YOLO model not found: {path}"
            return
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(path), opts, providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            LOGGER.info("YOLO grid detector loaded: %s", path.name)
        except Exception as e:
            self._load_error = f"Failed to load YOLO model: {e}"

    def _preprocess(
        self, image: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        """Letterbox resize + normalize for YOLO input.

        Returns (input_tensor, scale, pad_x, pad_y).
        """
        h, w = image.shape[:2]
        scale = min(self._imgsz / w, self._imgsz / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to square
        canvas = np.full(
            (self._imgsz, self._imgsz, 3), 114, dtype=np.uint8,
        )
        pad_x = (self._imgsz - new_w) / 2
        pad_y = (self._imgsz - new_h) / 2
        x0 = int(pad_x)
        y0 = int(pad_y)
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        # HWC BGR → CHW RGB, normalized to [0, 1]
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[None], scale, pad_x, pad_y

    def _parse_best_detection(
        self,
        preds: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        w_orig: int,
        h_orig: int,
    ) -> Optional[tuple[np.ndarray, float]]:
        """Parse YOLO output and return best (corners_4x2, confidence).

        Handles both (1, N, C) and (1, C, N) output layouts.
        """
        raw = np.squeeze(preds, axis=0)  # (N, C) or (C, N)

        # Determine layout: if dim0 < dim1, it's (C, N) → transpose
        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            raw = raw.T  # now (N, C)

        if raw.ndim != 2 or raw.shape[1] < 14:
            return None

        # Columns: cx, cy, w, h, conf (or class scores), kx1, ky1, ...
        # For single-class pose: [cx, cy, w, h, cls_score, kx1, ky1, kx2, ky2, ...]
        # Some exports: [cx, cy, w, h, obj_conf, cls_conf, kx1, ky1, ...]
        ncols = raw.shape[1]

        # End-to-end NMS format (1, max_det, 14):
        #   [x1, y1, x2, y2, conf, class_id, kx1, ky1, kx2, ky2, kx3, ky3, kx4, ky4]
        # Non-NMS format (1, 13, N) transposed to (N, 13):
        #   [cx, cy, w, h, cls_score, kx1, ky1, ..., kx4, ky4]
        if ncols >= 14:
            # End-to-end NMS: conf at col 4, class at col 5, kpts at col 6+
            conf = raw[:, 4]
            kpt_start = 6
            kpt_stride = 2
        elif ncols >= 13:
            # Non-NMS: cls_score at col 4, kpts at col 5+
            conf = raw[:, 4]
            kpt_start = 5
            kpt_stride = 2
        else:
            return None

        # Filter by confidence.
        mask = conf >= self._conf_threshold
        if not np.any(mask):
            return None

        # Best detection.
        indices = np.where(mask)[0]
        best_idx = indices[np.argmax(conf[mask])]
        best_conf = float(conf[best_idx])

        # Extract 4 keypoints.
        kpts = []
        for i in range(4):
            kx = float(raw[best_idx, kpt_start + i * kpt_stride])
            ky = float(raw[best_idx, kpt_start + i * kpt_stride + 1])
            # Map back from letterbox to original image coords.
            orig_x = (kx - pad_x) / scale
            orig_y = (ky - pad_y) / scale
            orig_x = max(0.0, min(orig_x, w_orig - 1))
            orig_y = max(0.0, min(orig_y, h_orig - 1))
            kpts.append([orig_x, orig_y])

        corners = np.float32(kpts)
        return corners, best_conf
