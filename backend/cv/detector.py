"""Grid detector abstractions: Protocol, result type, utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Result of grid detection: 4 corner coordinates + which detector found it."""

    corners: np.ndarray  # (4, 2), [TL, TR, BR, BL] in original image coords
    detector_name: str


@runtime_checkable
class GridDetector(Protocol):
    """Protocol for grid detectors."""

    def detect(self, image: np.ndarray) -> DetectionResult | None: ...


def warp_grid(image: np.ndarray, corners: np.ndarray, size: int = 450) -> np.ndarray:
    """Perspective-warp a grid region to a square image."""
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (size, size))


class ChainDetector:
    """Try multiple detectors in order, return the first successful result."""

    def __init__(self, detectors: list[GridDetector]) -> None:
        self._detectors = detectors

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        for d in self._detectors:
            result = d.detect(image)
            if result is not None:
                return result
        return None
