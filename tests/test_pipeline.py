"""Tests for OCR decoding: only a puzzle with a unique solution is trusted."""

from backend.ocr.pipeline import AMBIGUOUS, SOLVED, UNSOLVABLE, decode
from backend.solver.backtracking import SudokuSolver

PUZZLE = [
    [6, 0, 8, 0, 0, 3, 0, 2, 4],
    [4, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 8, 0],
    [8, 6, 0, 4, 7, 0, 0, 3, 0],
    [0, 7, 4, 1, 6, 2, 8, 9, 5],
    [1, 0, 0, 5, 0, 0, 0, 0, 7],
    [2, 0, 6, 0, 4, 0, 1, 0, 0],
    [0, 4, 3, 8, 0, 0, 6, 0, 0],
    [0, 8, 0, 7, 2, 6, 9, 0, 0],
]


def _metadata(grid):
    return {"cell_predictions": [
        {
            "index": idx, "row": idx // 9, "col": idx % 9,
            "value": grid[idx // 9][idx % 9], "confidence": 0.99,
            "candidates": [[grid[idx // 9][idx % 9] or 1, 0.01]],
        }
        for idx in range(81)
    ]}


def _with(grid, row, col, value):
    out = [r[:] for r in grid]
    out[row][col] = value
    return out


def test_unique_puzzle_is_solved_without_edits():
    result = decode(PUZZLE, _metadata(PUZZLE))
    assert result.status == SOLVED
    assert result.solution == SudokuSolver().solve(PUZZLE)
    assert result.corrections == []


def test_conflicting_misread_is_replaced_by_candidate():
    grid = _with(PUZZLE, 0, 0, 8)  # conflicts with r1c3
    meta = _metadata(grid)
    meta["cell_predictions"][0].update(confidence=0.35, candidates=[[8, 0.35], [6, 0.3]])

    result = decode(grid, meta)

    assert result.status == SOLVED
    assert result.grid == PUZZLE
    assert result.corrections == [{"row": 0, "col": 0, "from": 8, "to": 6}]


def test_valid_but_unsatisfiable_misread_is_replaced():
    grid = _with(PUZZLE, 0, 0, 7)  # locally valid, but no solution
    meta = _metadata(grid)
    meta["cell_predictions"][0].update(confidence=0.41, candidates=[[7, 0.41], [6, 0.4]])

    result = decode(grid, meta)

    assert result.grid == PUZZLE


def test_missed_given_is_restored_from_blank_candidate():
    solver = SudokuSolver()
    # Find a given whose removal makes the puzzle ambiguous.
    row, col = next(
        (r, c) for r in range(9) for c in range(9)
        if PUZZLE[r][c] and solver.count_solutions(_with(PUZZLE, r, c, 0)) > 1
    )
    grid = _with(PUZZLE, row, col, 0)
    meta = _metadata(grid)
    meta["cell_predictions"][row * 9 + col]["candidates"] = [[PUZZLE[row][col], 0.4]]

    result = decode(grid, meta)

    assert result.status == SOLVED
    assert result.grid == PUZZLE


def test_unrepairable_grids_report_why():
    sparse = [[0] * 9 for _ in range(9)]
    sparse[0][0] = 5
    assert decode(sparse, _metadata(sparse)).status == AMBIGUOUS

    broken = _with(PUZZLE, 0, 0, 7)
    result = decode(broken, _metadata(broken))
    assert result.status == UNSOLVABLE
    assert result.solution is None
