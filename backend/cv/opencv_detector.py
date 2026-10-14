"""OpenCV contour-based grid detector, wrapped as GridDetector."""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from .detector import DetectionResult
from .grid_detector import find_grid_with_corners, has_regular_grid_lines

# Thresholding and morphology use fixed pixel sizes, so no single scale suits
# every image: on a 12MP photo of a screen they pick up moiré and lose the grid
# outline, some paper photos lose thin lines when downscaled, and in images
# narrower than this the opening erases 1px lines.
DETECT_WIDTH = 1280


class OpenCVGridDetector:
    """Detect sudoku grids using OpenCV contour detection.

    Tries the input, then a copy DETECT_WIDTH wide, each directly
    and with CLAHE. The first warp with regular grid lines wins; otherwise the
    first irregular candidate is returned.
    """

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        fallback = None
        for variant, scale in _variants(image):
            result = find_grid_with_corners(variant)
            if result is None:
                continue
            warped, corners = result
            corners = corners / scale
            if has_regular_grid_lines(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)):
                return DetectionResult(corners=corners, detector_name="opencv")
            if fallback is None:
                fallback = DetectionResult(corners=corners, detector_name="opencv")
        return fallback


def _variants(image: np.ndarray) -> Iterator[tuple[np.ndarray, float]]:
    scales = dict.fromkeys([1.0, DETECT_WIDTH / image.shape[1]])
    for enhance in (False, True):
        for scale in scales:
            img = image if scale == 1.0 else cv2.resize(
                image, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            if enhance:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
                img = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
            yield img, scale
