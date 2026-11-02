"""Sudoku solver: bitmask backtracking with the MRV heuristic.

Grids are 9x9 (values 1-9) or 16x16 (values 1-16, shown as 1-9 and A-G);
0 is an empty cell. The box side is the square root of the grid size.
"""

from __future__ import annotations

import math

Grid = list[list[int]]

SIZES = (9, 16)


def box_side(size: int) -> int:
    return math.isqrt(size)


class SudokuSolver:
    """Solves Sudoku puzzles using backtracking."""

    def __init__(self, max_nodes: int = 100_000):
        self.max_nodes = max_nodes

    def solve(self, grid: Grid) -> Grid | None:
        """Return the first solution found, or None."""
        found, _ = _search(grid, 1, self.max_nodes)
        return found[0] if found else None

    def count_solutions(self, grid: Grid, max_count: int = 2) -> int:
        """Count solutions up to max_count.

        An exhausted node budget counts as max_count: uniqueness was not proven.
        """
        found, exhausted = _search(grid, max_count, self.max_nodes)
        return max_count if exhausted else len(found)

    def solve_unique(self, grid: Grid) -> tuple[Grid | None, int]:
        """Return (solution if unique else None, solution count capped at 2)."""
        found, exhausted = _search(grid, 2, self.max_nodes)
        count = 2 if exhausted else len(found)
        return (found[0] if count == 1 else None), count


class _BudgetExhausted(Exception):
    pass


def _search(grid: Grid, limit: int, max_nodes: int) -> tuple[list[Grid], bool]:
    """Find up to `limit` solutions. Returns (solutions, budget_exhausted)."""
    size = len(grid)
    box = box_side(size)
    all_values = ((1 << size) - 1) << 1  # bits 1..size
    rows, cols, boxes = [0] * size, [0] * size, [0] * size
    empties: list[tuple[int, int, int]] = []
    cells = [list(row) for row in grid]
    for r in range(size):
        for c in range(size):
            b = (r // box) * box + c // box
            value = cells[r][c]
            if value == 0:
                empties.append((r, c, b))
                continue
            bit = 1 << value
            if (rows[r] | cols[c] | boxes[b]) & bit:
                return [], False
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

    found: list[Grid] = []
    nodes = 0

    def recurse() -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes:
            raise _BudgetExhausted
        if not empties:
            found.append([row[:] for row in cells])
            return len(found) >= limit

        # Branch on the most constrained choice: a cell with the fewest
        # candidates (MRV), or a digit with only one place left in a unit.
        best_i, best_mask, best_n = 0, 0, size + 1
        masks = []
        for i, (r, c, b) in enumerate(empties):
            mask = all_values & ~(rows[r] | cols[c] | boxes[b])
            masks.append(mask)
            n = mask.bit_count()
            if n < best_n:
                best_i, best_mask, best_n = i, mask, n
                if n <= 1:
                    break
        if best_n == 0:
            return False
        if best_n > 1:
            for unit, used in enumerate((rows, cols, boxes)):
                groups: dict[int, list[int]] = {}
                for i, cell in enumerate(empties):
                    groups.setdefault(cell[unit], []).append(i)
                for u, members in groups.items():
                    missing = all_values & ~used[u]
                    while missing:
                        bit = missing & -missing
                        missing ^= bit
                        places = [i for i in members if masks[i] & bit]
                        if not places:
                            return False
                        if len(places) == 1:
                            best_i, best_mask, best_n = places[0], bit, 1
                            break
                    if best_n == 1:
                        break
                if best_n == 1:
                    break

        cell = empties[best_i]
        r, c, b = cell
        empties[best_i] = empties[-1]
        empties.pop()
        mask = best_mask
        done = False
        while mask and not done:
            bit = mask & -mask
            mask ^= bit
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit
            cells[r][c] = bit.bit_length() - 1
            done = recurse()
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
        cells[r][c] = 0
        empties.append(cell)
        empties[best_i], empties[-1] = empties[-1], empties[best_i]
        return done

    try:
        recurse()
    except _BudgetExhausted:
        return found, True
    return found, False


def solve(grid: Grid) -> Grid | None:
    """Convenience function to solve a Sudoku grid."""
    return SudokuSolver().solve(grid)


def is_valid_grid(grid: Grid) -> bool:
    """Check that a 9x9 or 16x16 grid of 0..size has no duplicate givens."""
    n = len(grid)
    if n not in SIZES or any(len(row) != n for row in grid):
        return False
    if any(not 0 <= cell <= n for row in grid for cell in row):
        return False
    return all(
        is_valid_placement(grid, r, c, grid[r][c]) for r in range(n) for c in range(n)
    )


def is_valid_placement(grid: Grid, row: int, col: int, val: int) -> bool:
    """Check whether placing *val* at (row, col) conflicts with existing values.

    Returns True for val == 0 (empty cell).
    """
    if val == 0:
        return True
    n = len(grid)

    for c in range(n):
        if c != col and grid[row][c] == val:
            return False

    for r in range(n):
        if r != row and grid[r][col] == val:
            return False

    box = box_side(n)
    box_row = (row // box) * box
    box_col = (col // box) * box
    for r in range(box_row, box_row + box):
        for c in range(box_col, box_col + box):
            if (r != row or c != col) and grid[r][c] == val:
                return False

    return True
