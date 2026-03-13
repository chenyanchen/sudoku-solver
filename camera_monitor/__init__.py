"""Camera-based realtime Sudoku solver using macOS Continuity Camera."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ProcessingResult:
    corners: Optional[np.ndarray] = None        # 4x2 in original frame coords
    warped: Optional[np.ndarray] = None          # 450x450 BGR warped grid
    original_grid: Optional[list[list[int]]] = None  # 9x9 OCR result
    solved_grid: Optional[list[list[int]]] = None    # 9x9 solved result
    hints: Optional[list[dict[str, Any]]] = None     # positioned in frame coords
    givens: int = 0
