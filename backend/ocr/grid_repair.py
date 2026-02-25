"""Repair OCR grids with CNN candidate hints before solving."""

from __future__ import annotations

import copy
from typing import Any

from ..solver.backtracking import SudokuSolver, is_valid_grid, is_valid_placement


def try_repair_grid_with_candidates(
    grid: list[list[int]],
    cell_predictions: list[dict[str, Any]] | None,
    max_changes: int = 2,
    max_cells: int = 14,
    min_candidate_prob: float = 0.15,
) -> tuple[list[list[int]], list[list[int]] | None, dict[str, Any]]:
    """Try a bounded search over low-confidence cells and solve the repaired grid."""
    base_grid = copy.deepcopy(grid)
    solver = SudokuSolver()
    solved = solver.solve(base_grid)
    if solved is not None:
        return base_grid, solved, {"attempted": False, "changes": 0, "reason": "solved"}

    if not cell_predictions or len(cell_predictions) != 81:
        return base_grid, None, {
            "attempted": False,
            "changes": 0,
            "reason": "missing_predictions",
        }

    max_changes = max(1, int(max_changes))
    max_cells = max(1, int(max_cells))
    min_candidate_prob = max(0.0, min(1.0, float(min_candidate_prob)))

    selected = _select_repair_cells(base_grid, cell_predictions, max_cells=max_cells)
    if not selected:
        return base_grid, None, {
            "attempted": False,
            "changes": 0,
            "reason": "no_repair_cells",
        }

    for budget in range(1, max_changes + 1):
        work_grid = copy.deepcopy(base_grid)
        changes: dict[int, tuple[int, int]] = {}
        result = _search_repair(
            grid=work_grid,
            original_grid=base_grid,
            solver=solver,
            cell_predictions=cell_predictions,
            selected_indices=selected,
            min_candidate_prob=min_candidate_prob,
            budget=budget,
            pos=0,
            changes=changes,
        )
        if result is not None:
            repaired_grid, solved_grid, changed_cells = result
            return repaired_grid, solved_grid, {
                "attempted": True,
                "changes": len(changed_cells),
                "budget": budget,
                "changed_cells": [
                    {"index": idx, "from": old, "to": new}
                    for idx, (old, new) in sorted(changed_cells.items())
                ],
            }

    return base_grid, None, {
        "attempted": True,
        "changes": 0,
        "reason": "no_candidate_solution",
        "selected_cells": len(selected),
    }


def _search_repair(
    grid: list[list[int]],
    original_grid: list[list[int]],
    solver: SudokuSolver,
    cell_predictions: list[dict[str, Any]],
    selected_indices: list[int],
    min_candidate_prob: float,
    budget: int,
    pos: int,
    changes: dict[int, tuple[int, int]],
) -> tuple[list[list[int]], list[list[int]], dict[int, tuple[int, int]]] | None:
    if len(changes) > budget:
        return None

    if pos >= len(selected_indices):
        if not changes:
            return None
        if not is_valid_grid(copy.deepcopy(grid)):
            return None

        solved = solver.solve(grid)
        if solved is None:
            return None
        return copy.deepcopy(grid), solved, dict(changes)

    idx = selected_indices[pos]
    row, col = divmod(idx, 9)
    original_value = original_grid[row][col]
    current_value = grid[row][col]

    options = _build_repair_options(
        current_value=current_value,
        prediction=cell_predictions[idx],
        min_candidate_prob=min_candidate_prob,
    )

    for option in options:
        changed = option != original_value
        if changed and len(changes) >= budget:
            continue

        if option != 0 and not is_valid_placement(grid, row, col, option):
            continue

        prev_value = grid[row][col]
        prev_change = changes.get(idx)

        grid[row][col] = option
        if option != original_value:
            changes[idx] = (original_value, option)
        else:
            changes.pop(idx, None)

        result = _search_repair(
            grid=grid,
            original_grid=original_grid,
            solver=solver,
            cell_predictions=cell_predictions,
            selected_indices=selected_indices,
            min_candidate_prob=min_candidate_prob,
            budget=budget,
            pos=pos + 1,
            changes=changes,
        )
        if result is not None:
            return result

        grid[row][col] = prev_value
        if prev_change is None:
            changes.pop(idx, None)
        else:
            changes[idx] = prev_change

    return None


def _build_repair_options(
    current_value: int,
    prediction: dict[str, Any],
    min_candidate_prob: float,
    max_options: int = 4,
) -> list[int]:
    options: list[int] = []
    seen: set[int] = set()

    def add(value: int) -> None:
        if value in seen:
            return
        seen.add(value)
        options.append(value)

    add(int(current_value))

    candidates = prediction.get("candidates", [])
    if isinstance(candidates, list):
        for entry in candidates:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            try:
                digit = int(entry[0])
                prob = float(entry[1])
            except (TypeError, ValueError):
                continue
            if not (1 <= digit <= 9):
                continue
            if prob < min_candidate_prob:
                continue
            add(digit)
            if len(options) >= max_options:
                break

    if current_value != 0:
        add(0)

    return options[:max_options]


def _select_repair_cells(
    grid: list[list[int]],
    cell_predictions: list[dict[str, Any]],
    max_cells: int,
) -> list[int]:
    conflicts = _find_conflict_cells(grid)
    selected: list[int] = []

    def confidence(index: int) -> float:
        pred = cell_predictions[index]
        raw = pred.get("confidence", 1.0) if isinstance(pred, dict) else 1.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 1.0

    for idx in sorted(conflicts, key=confidence):
        selected.append(idx)
        if len(selected) >= max_cells:
            return selected

    uncertain_nonzero: list[tuple[float, int]] = []
    for idx in range(81):
        if idx in conflicts:
            continue
        row, col = divmod(idx, 9)
        if grid[row][col] == 0:
            continue
        uncertain_nonzero.append((confidence(idx), idx))

    uncertain_nonzero.sort(key=lambda item: item[0])
    for _, idx in uncertain_nonzero:
        selected.append(idx)
        if len(selected) >= max_cells:
            break

    return selected


def _find_conflict_cells(grid: list[list[int]]) -> set[int]:
    conflicts: set[int] = set()
    for row in range(9):
        for col in range(9):
            val = int(grid[row][col])
            if val == 0:
                continue
            if not is_valid_placement(grid, row, col, val):
                conflicts.add(row * 9 + col)
    return conflicts
