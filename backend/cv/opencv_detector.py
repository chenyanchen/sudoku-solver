"""OpenCV contour-based grid detector, wrapped as GridDetector."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import DetectionResult
from .grid_detector import find_quads, snap_quads

# Thresholding and morphology use fixed pixel sizes, so no single scale suits
# every image: on a 12MP photo of a screen they pick up moiré and lose the grid
# outline, some paper photos lose thin lines when downscaled, and in images
# narrower than this the opening erases 1px lines.
DETECT_WIDTH = 1280


class OpenCVGridDetector:
    """Detect sudoku grids using OpenCV contour detection.

    Tries the input, then a copy DETECT_WIDTH wide, each directly and with
    CLAHE. The first warp with regular grid lines wins. Only when no variant
    has one are the irregular quads snapped to the lattice around them, in the
    same order; failing that, the first irregular quad is returned.
    """

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        # Fitting a lattice costs ~100x checking a quad: on a 12MP photo with
        # moiré, snapping the full-resolution quads took 1 s while the 1280px
        # copy had the grid outright.
        pending = []
        for scale, enhance in _variants(image):
            grids, fallbacks = find_quads(_variant(image, scale, enhance))
            if grids:
                return DetectionResult(corners=grids[0][1] / scale, detector_name="opencv")
            if fallbacks:
                pending.append((scale, enhance, [corners for _, corners in fallbacks]))
        # Each variant is made again rather than kept: at full resolution
        # each is a copy of the image.
        for scale, enhance, quads in pending:
            snapped = snap_quads(_variant(image, scale, enhance), quads)
            if snapped:
                return DetectionResult(corners=snapped[0][1] / scale, detector_name="opencv")
        if not pending:
            return None
        scale, _, quads = pending[0]
        return DetectionResult(corners=quads[0] / scale, detector_name="opencv")


def _variants(image: np.ndarray) -> list[tuple[float, bool]]:
    scales = dict.fromkeys([1.0, DETECT_WIDTH / image.shape[1]])
    return [(scale, enhance) for enhance in (False, True) for scale in scales]


def _variant(image: np.ndarray, scale: float, enhance: bool) -> np.ndarray:
    img = image if scale == 1.0 else cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    if enhance:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        img = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    return img
