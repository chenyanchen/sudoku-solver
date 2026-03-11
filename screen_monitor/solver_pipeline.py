"""OCR + solve pipeline for a single warped grid image."""

from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

from backend.cv.cell_extractor import extract_cells
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.ocr.grid_repair import try_repair_grid_with_candidates
from backend.solver.backtracking import SudokuSolver


def _prepare_grid_for_screen_ocr(grid_img: np.ndarray) -> np.ndarray:
    """Screen-optimized preprocessing: skip CLAHE (uniform lighting)."""
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY) if len(grid_img.shape) == 3 else grid_img
    if np.mean(gray) <= 128:
        gray = cv2.bitwise_not(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def ocr_grid_from_warped(
    warped_bgr: Any,
    reader: CnnDigitReader,
    min_givens: int,
) -> tuple[Optional[dict[str, Any]], str, int]:
    """Run OCR on a warped 450x450 grid image (no solving).

    Returns (ocr_result_or_None, reason_string, given_count).
    ocr_result contains: original_grid, metadata, _timing.
    """
    t0 = time.perf_counter()
    ocr_grid = _prepare_grid_for_screen_ocr(warped_bgr)
    cells = extract_cells(ocr_grid)
    if len(cells) != 81:
        return None, "bad-cells", 0

    t1 = time.perf_counter()
    original_grid, metadata = reader.recognize_grid_with_metadata(
        cells, threshold=None,
    )
    t2 = time.perf_counter()
    given_cells = sum(1 for row in original_grid for cell in row if cell != 0)
    if given_cells < min_givens:
        return None, f"low-givens:{given_cells}", given_cells

    return {
        "original_grid": original_grid,
        "metadata": metadata,
        "givens": given_cells,
        "_timing": {
            "extract_cells_ms": (t1 - t0) * 1000.0,
            "ocr_ms": (t2 - t1) * 1000.0,
        },
    }, "ok", given_cells


def solve_from_grid(
    ocr_result: dict[str, Any],
) -> tuple[Optional[dict[str, Any]], str]:
    """Solve a grid from OCR results.

    Returns (payload_dict_or_None, reason_string).
    """
    original_grid = ocr_result["original_grid"]
    metadata = ocr_result["metadata"]
    given_cells = ocr_result["givens"]

    t0 = time.perf_counter()
    solver = SudokuSolver()
    solved_grid = solver.solve(original_grid)

    if solved_grid is None:
        predictions = metadata.get("cell_predictions")
        repaired_grid, repaired_solution, _ = try_repair_grid_with_candidates(
            grid=original_grid,
            cell_predictions=predictions if isinstance(predictions, list) else None,
            max_changes=4,
            max_cells=20,
        )
        if repaired_solution is not None:
            original_grid = repaired_grid
            solved_grid = repaired_solution
    t1 = time.perf_counter()

    if solved_grid is None:
        return None, f"unsolved:{given_cells}"

    hints = []
    for row in range(9):
        for col in range(9):
            if original_grid[row][col] == 0:
                hints.append({
                    "row": row, "col": col,
                    "digit": int(solved_grid[row][col]),
                })

    if not hints:
        return None, f"no-hints:{given_cells}"

    # Merge OCR timing with solve timing.
    timing = dict(ocr_result.get("_timing", {}))
    timing["solve_ms"] = (t1 - t0) * 1000.0

    return {
        "original_grid": original_grid,
        "solved_grid": solved_grid,
        "hints": hints,
        "givens": given_cells,
        "metadata": metadata,
        "_timing": timing,
    }, "ok"


def solve_hints_from_warped(
    warped_bgr: Any,
    reader: CnnDigitReader,
    min_givens: int,
) -> tuple[Optional[dict[str, Any]], str, int]:
    """Run OCR + solve on a warped 450x450 grid image.

    Returns (payload_dict_or_None, reason_string, given_count).
    """
    ocr_result, reason, given_cells = ocr_grid_from_warped(
        warped_bgr, reader, min_givens,
    )
    if ocr_result is None:
        return None, reason, given_cells

    payload, solve_reason = solve_from_grid(ocr_result)
    if payload is None:
        return None, solve_reason, given_cells

    return payload, "ok", given_cells
