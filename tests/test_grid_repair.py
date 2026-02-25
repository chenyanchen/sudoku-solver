"""Tests for OCR grid auto-repair using CNN candidate hints."""

from backend.ocr.grid_repair import try_repair_grid_with_candidates


def _build_default_predictions(grid: list[list[int]]) -> list[dict]:
    preds = []
    for idx in range(81):
        row, col = divmod(idx, 9)
        value = int(grid[row][col])
        preds.append(
            {
                "index": idx,
                "row": row,
                "col": col,
                "value": value,
                "confidence": 0.99,
                "candidates": [[value, 0.99]] if value != 0 else [],
            }
        )
    return preds


def test_repair_recovers_single_conflict_cell():
    grid = [
        [8, 0, 8, 0, 0, 3, 0, 2, 4],  # r1c1 should be 6, now conflicts with r1c3
        [4, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 8, 0],
        [8, 6, 0, 4, 7, 0, 0, 3, 0],
        [0, 7, 4, 1, 6, 2, 8, 9, 5],
        [1, 0, 0, 5, 0, 0, 0, 0, 7],
        [2, 0, 6, 0, 4, 0, 1, 0, 0],
        [0, 4, 3, 8, 0, 0, 6, 0, 0],
        [0, 8, 0, 7, 2, 6, 9, 0, 0],
    ]
    preds = _build_default_predictions(grid)
    preds[0]["confidence"] = 0.35
    preds[0]["candidates"] = [[6, 0.86], [8, 0.12]]

    repaired_grid, solved_grid, info = try_repair_grid_with_candidates(
        grid, preds, max_changes=1, max_cells=1
    )

    assert solved_grid is not None
    assert repaired_grid[0][0] in (0, 6)
    assert info["changes"] == 1


def test_repair_recovers_valid_but_unsatisfiable_grid():
    grid = [
        [7, 0, 8, 0, 0, 3, 0, 2, 4],  # r1c1 should be 6, but 7 is locally valid
        [4, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 8, 0],
        [8, 6, 0, 4, 7, 0, 0, 3, 0],
        [0, 7, 4, 1, 6, 2, 8, 9, 5],
        [1, 0, 0, 5, 0, 0, 0, 0, 7],
        [2, 0, 6, 0, 4, 0, 1, 0, 0],
        [0, 4, 3, 8, 0, 0, 6, 0, 0],
        [0, 8, 0, 7, 2, 6, 9, 0, 0],
    ]
    preds = _build_default_predictions(grid)
    preds[0]["confidence"] = 0.41
    preds[0]["candidates"] = [[6, 0.79], [7, 0.18]]

    repaired_grid, solved_grid, info = try_repair_grid_with_candidates(
        grid, preds, max_changes=1, max_cells=10
    )

    assert solved_grid is not None
    assert repaired_grid[0][0] == 6
    assert info["changes"] == 1
