"""Sudoku solver: bitmask backtracking with the MRV heuristic."""

from __future__ import annotations

Grid = list[list[int]]

_ALL_DIGITS = 0b1111111110  # bits 1..9


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

    def common_cells(self, grid: Grid, max_solutions: int = 100) -> tuple[Grid, int] | None:
        """For a puzzle with 2..max_solutions solutions, the cells all of them share.

        Returns (grid with 0 where solutions differ, solution count), or None
        when there are fewer than 2 solutions or they could not all be found.
        A missed given only adds solutions, so shared cells hold in the true
        puzzle too.
        """
        found, exhausted = _search(grid, max_solutions + 1, self.max_nodes)
        if exhausted or not 2 <= len(found) <= max_solutions:
            return None
        shared = [
            [found[0][r][c] if all(s[r][c] == found[0][r][c] for s in found) else 0 for c in range(9)]
            for r in range(9)
        ]
        return shared, len(found)


class _BudgetExhausted(Exception):
    pass


def _search(grid: Grid, limit: int, max_nodes: int) -> tuple[list[Grid], bool]:
    """Find up to `limit` solutions. Returns (solutions, budget_exhausted)."""
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    empties: list[tuple[int, int, int]] = []
    cells = [list(row) for row in grid]
    for r in range(9):
        for c in range(9):
            b = (r // 3) * 3 + c // 3
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
        best_i, best_mask, best_n = 0, 0, 10
        masks = []
        for i, (r, c, b) in enumerate(empties):
            mask = _ALL_DIGITS & ~(rows[r] | cols[c] | boxes[b])
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
                    missing = _ALL_DIGITS & ~used[u]
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
    """Check that a 9x9 grid of 0..9 has no duplicate givens."""
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        return False
    if any(not 0 <= cell <= 9 for row in grid for cell in row):
        return False
    return all(
        is_valid_placement(grid, r, c, grid[r][c]) for r in range(9) for c in range(9)
    )


def is_valid_placement(grid: Grid, row: int, col: int, val: int) -> bool:
    """Check whether placing *val* at (row, col) conflicts with existing values.

    Returns True for val == 0 (empty cell).
    """
    if val == 0:
        return True

    for c in range(9):
        if c != col and grid[row][c] == val:
            return False

    for r in range(9):
        if r != row and grid[r][col] == val:
            return False

    box_row = (row // 3) * 3
    box_col = (col // 3) * 3
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if (r != row or c != col) and grid[r][c] == val:
                return False

    return True
