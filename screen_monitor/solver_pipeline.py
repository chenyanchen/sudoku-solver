"""OCR + solve pipeline for a single warped grid image."""

from __future__ import annotations

import time
from typing import Any, Optional

from backend.api.routes import prepare_grid_for_ocr
from backend.cv.cell_extractor import extract_cells
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.ocr.grid_repair import try_repair_grid_with_candidates
from backend.solver.backtracking import SudokuSolver


def solve_hints_from_warped(
    warped_bgr: "Any",
    reader: CnnDigitReader,
    min_givens: int,
) -> tuple[Optional[dict[str, Any]], str, int]:
    """Run OCR + solve on a warped 450x450 grid image.

    Returns (payload_dict_or_None, reason_string, given_count).
    """
    t0 = time.perf_counter()
    ocr_grid = prepare_grid_for_ocr(warped_bgr)
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
    t3 = time.perf_counter()

    if solved_grid is None:
        return None, f"unsolved:{given_cells}", given_cells

    hints = []
    for row in range(9):
        for col in range(9):
            if original_grid[row][col] == 0:
                hints.append({
                    "row": row, "col": col,
                    "digit": int(solved_grid[row][col]),
                })

    if not hints:
        return None, f"no-hints:{given_cells}", given_cells

    return {
        "original_grid": original_grid,
        "solved_grid": solved_grid,
        "hints": hints,
        "givens": given_cells,
        "metadata": metadata,
        "_timing": {
            "extract_cells_ms": (t1 - t0) * 1000.0,
            "ocr_ms": (t2 - t1) * 1000.0,
            "solve_ms": (t3 - t2) * 1000.0,
        },
    }, "ok", given_cells
