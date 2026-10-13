"""Warped grid → OCR → decode → unique solution, shared by every entry point.

A recognized puzzle is trusted only when its givens have exactly one solution.
Missed givens leave a valid but ambiguous puzzle; misread givens leave an
unsolvable one. Both are corrected with a bounded search over CNN candidates.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..cv.cell_extractor import extract_cells
from ..solver.backtracking import Grid, SudokuSolver, is_valid_placement
from .cnn_digit_reader import CnnDigitReader

SOLVED, AMBIGUOUS, UNSOLVABLE = "solved", "ambiguous", "unsolvable"


@dataclass
class Recognition:
    grid: Grid  # givens after correction
    solution: Grid | None  # set only when the givens have exactly one solution
    status: str
    metadata: dict[str, Any]
    corrections: list[dict[str, int]] = field(default_factory=list)
    timing: dict[str, float] = field(default_factory=dict)

    @property
    def givens(self) -> int:
        return sum(value != 0 for row in self.grid for value in row)

    @property
    def hints(self) -> list[dict[str, int]]:
        if self.solution is None:
            return []
        return [
            {"row": r, "col": c, "digit": self.solution[r][c]}
            for r in range(9) for c in range(9) if self.grid[r][c] == 0
        ]


def prepare_grid_for_ocr(grid_img: np.ndarray) -> np.ndarray:
    """Binarize a warped grid against local lighting; returns BGR for cell extraction."""
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) <= 128:
        gray = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def recognize(
    reader: CnnDigitReader, warped: np.ndarray, max_changes: int = 2, max_cells: int = 10,
) -> Recognition:
    """Recognize and solve a 450x450 warped grid."""
    timing = {"ocr_ms": 0.0, "solve_ms": 0.0}

    def timed(key: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        timing[key] += (time.perf_counter() - t0) * 1000.0
        return out

    ocr_grid = prepare_grid_for_ocr(warped)
    grid, metadata = timed("ocr_ms", reader.recognize_grid_with_metadata, extract_cells(ocr_grid))
    result = timed("solve_ms", decode, grid, metadata, max_changes, max_cells)

    if result.status != SOLVED:
        # Thick screen grid lines survive the normal 1px cell border crop.
        centered = [
            ocr_grid[r * 50 + 10 : (r + 1) * 50 - 10, c * 50 + 10 : (c + 1) * 50 - 10]
            for r in range(9) for c in range(9)
        ]
        grid, metadata = timed(
            "ocr_ms", reader.recognize_grid_with_metadata, centered, threshold=0.3,
        )
        retry = timed("solve_ms", decode, grid, metadata, max_changes, max_cells)
        if retry.status == SOLVED:
            result = retry

    result.timing = timing
    return result


def decode(
    grid: Grid,
    metadata: dict[str, Any],
    max_changes: int = 2,
    max_cells: int = 10,
    min_prob: float = 0.15,
) -> Recognition:
    """Return the most plausible correction of `grid` that has a unique solution.

    Tries every combination of up to `max_changes` edits among the `max_cells`
    most doubtful cells, fewest edits first.
    """
    solver = SudokuSolver()
    solution, count = solver.solve_unique(grid)
    if count == 1:
        return Recognition(grid, solution, SOLVED, metadata)

    # Ambiguous → a given was missed; unsolvable → a given was misread.
    fill_blanks = count > 1
    givens = sum(value != 0 for row in grid for value in row)
    if fill_blanks and givens + max_changes < 17:  # no unique puzzle has < 17 givens
        return Recognition(grid, None, AMBIGUOUS, metadata)

    predictions = metadata["cell_predictions"]
    options = _edit_options(grid, predictions, fill_blanks, min_prob)
    doubtful = sorted(options, key=lambda idx: -options[idx][0])[:max_cells]

    for k in range(1, max_changes + 1):
        for cells in itertools.combinations(doubtful, k):
            for values in itertools.product(*(options[idx][1] for idx in cells)):
                candidate = [row[:] for row in grid]
                for idx, value in zip(cells, values):
                    candidate[idx // 9][idx % 9] = value
                if not all(
                    is_valid_placement(candidate, idx // 9, idx % 9, value)
                    for idx, value in zip(cells, values)
                ):
                    continue
                solution, count = solver.solve_unique(candidate)
                if count == 1:
                    corrections = [
                        {"row": idx // 9, "col": idx % 9, "from": grid[idx // 9][idx % 9], "to": value}
                        for idx, value in zip(cells, values)
                    ]
                    return Recognition(candidate, solution, SOLVED, metadata, corrections)

    return Recognition(grid, None, AMBIGUOUS if fill_blanks else UNSOLVABLE, metadata)


def _edit_options(
    grid: Grid, predictions: list[dict[str, Any]], fill_blanks: bool, min_prob: float,
) -> dict[int, tuple[float, list[int]]]:
    """Map cell index → (doubt, alternative values) for cells worth editing."""
    options: dict[int, tuple[float, list[int]]] = {}
    for pred in predictions:
        idx, value = pred["index"], grid[pred["row"]][pred["col"]]
        digits = [d for d, p in pred["candidates"] if p >= min_prob and d != value]
        if fill_blanks and value == 0 and digits:
            options[idx] = (pred["candidates"][0][1], digits)
        elif not fill_blanks and value != 0:
            conflict = not is_valid_placement(grid, pred["row"], pred["col"], value)
            doubt = 2.0 if conflict else 1.0 - pred["confidence"]
            options[idx] = (doubt, digits + [0])
    return options
