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

_CELL = 48  # cell side after the default 1px margin
# Median digit height / cell side is 0.48–0.69 across the training grids;
# screenshots with small fonts (~0.35–0.40) are read poorly until zoomed in.
_SMALL_DIGIT_HEIGHT = 0.45
_TRAIN_DIGIT_HEIGHT = 0.55
_ZOOMED_DIGIT_THRESHOLD = 0.3
# Below this blank probability a cell read as blank holds a digit in a font
# the CNN is unsure of; across the labeled grids every such cell was a digit.
_INKED_BLANK_PROB = 0.05


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


def read_grid(reader: CnnDigitReader, warped: np.ndarray) -> tuple[Grid, dict[str, Any]]:
    """Raw CNN reading of a warped grid (no Sudoku constraints applied)."""
    cells, zoomed = ocr_cells(prepare_grid_for_ocr(warped))
    # Zoomed digits are outside the training distribution: accept lower
    # confidence and rely on decode()'s uniqueness check to reject misreads.
    return reader.recognize_grid_with_metadata(
        cells, threshold=_ZOOMED_DIGIT_THRESHOLD if zoomed else None,
    )


def ocr_cells(ocr_grid: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """Cut 81 cells, zooming in when digits are smaller than in the training data.

    Returns (cells, zoomed). Training and inference must both use this.
    """
    heights = _digit_heights(ocr_grid)
    if len(heights) < 5:
        return extract_cells(ocr_grid), False
    height = float(np.median(heights))
    if height / _CELL >= _SMALL_DIGIT_HEIGHT:
        return extract_cells(ocr_grid), False
    margin = round((50 - height / _TRAIN_DIGIT_HEIGHT) / 2)
    return extract_cells(ocr_grid, margin), True


def _digit_heights(ocr_grid: np.ndarray) -> list[int]:
    """Height of the tallest interior ink blob in each cell that has one."""
    gray = cv2.cvtColor(ocr_grid, cv2.COLOR_BGR2GRAY)
    heights = []
    for cell in extract_cells(gray):
        count, _, stats, _ = cv2.connectedComponentsWithStats((cell < 128).astype(np.uint8))
        tallest = max(
            (h for x, y, w, h, _ in stats[1:count]
             if x > 0 and y > 0 and x + w < _CELL and y + h < _CELL),  # skip grid-line fragments
            default=0,
        )
        if tallest >= _CELL * 0.2:
            heights.append(tallest)
    return heights


def recognize(
    reader: CnnDigitReader, warped: np.ndarray, max_changes: int = 2, max_cells: int = 10,
) -> Recognition:
    """Recognize and solve a 450x450 warped grid."""
    t0 = time.perf_counter()
    grid, metadata = read_grid(reader, warped)
    t1 = time.perf_counter()
    result = decode(grid, metadata, max_changes, max_cells)
    t2 = time.perf_counter()
    result.timing = {"ocr_ms": (t1 - t0) * 1000.0, "solve_ms": (t2 - t1) * 1000.0}
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
    most doubtful cells, fewest edits first. Cells that are surely not blank
    start with their top candidate, so only a misread among them costs an edit.
    A correction is trusted only when no other with as few edits leads to a
    different solution: then the CNN cannot tell which givens were misread.
    """
    grid, metadata = _fill_inked_blanks(grid, metadata)
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

    found: Recognition | None = None
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
                if count != 1:
                    continue
                if found is not None:
                    if solution != found.solution:
                        return Recognition(grid, None, AMBIGUOUS if fill_blanks else UNSOLVABLE, metadata)
                    continue
                corrections = [
                    {"row": idx // 9, "col": idx % 9, "from": grid[idx // 9][idx % 9], "to": value}
                    for idx, value in zip(cells, values)
                ]
                found = Recognition(candidate, solution, SOLVED, metadata, corrections)
        if found is not None:
            return found

    return Recognition(grid, None, AMBIGUOUS if fill_blanks else UNSOLVABLE, metadata)


def _fill_inked_blanks(
    grid: Grid, metadata: dict[str, Any],
) -> tuple[Grid, dict[str, Any]]:
    """Give blank-read cells with ink their top candidate, at that candidate's confidence."""
    grid = [row[:] for row in grid]
    predictions = []
    for pred in metadata["cell_predictions"]:
        if pred["value"] == 0 and pred["confidence"] < _INKED_BLANK_PROB and pred["candidates"]:
            digit, prob = pred["candidates"][0]
            grid[pred["row"]][pred["col"]] = digit
            pred = {**pred, "value": digit, "confidence": prob}
        predictions.append(pred)
    return grid, {**metadata, "cell_predictions": predictions}


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
