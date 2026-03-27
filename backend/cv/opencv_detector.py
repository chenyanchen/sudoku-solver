"""OpenCV contour-based grid detector, wrapped as GridDetector."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import DetectionResult
from .grid_detector import find_grid_with_corners


class OpenCVGridDetector:
    """Detect sudoku grids using OpenCV contour detection.

    Tries direct detection first, then retries with CLAHE enhancement.
    """

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        # Direct detection.
        result = find_grid_with_corners(image)
        if result is not None:
            _, corners = result
            return DetectionResult(corners=corners, detector_name="opencv")

        # CLAHE-enhanced retry.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        result = find_grid_with_corners(enhanced)
        if result is not None:
            _, corners = result
            return DetectionResult(corners=corners, detector_name="opencv")

        return None
