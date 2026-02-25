"""Sudoku solver using backtracking algorithm."""

from typing import Optional, List, Tuple
import copy


Grid = List[List[int]]


class SudokuSolver:
    """Solves Sudoku puzzles using backtracking."""

    def __init__(self):
        self.solutions_count = 0

    def solve(self, grid: Grid) -> Optional[Grid]:
        """
        Solve a Sudoku puzzle.

        Args:
            grid: 9x9 list of lists with 0 for empty cells

        Returns:
            Solved 9x9 grid if solution exists, None otherwise
        """
        self.solutions_count = 0
        grid_copy = copy.deepcopy(grid)
        if not self._is_consistent_grid(grid_copy):
            return None
        if self._solve_recursive(grid_copy):
            return grid_copy
        return None

    def _solve_recursive(self, grid: Grid) -> bool:
        """Recursively solve the puzzle using backtracking."""
        empty = self._find_empty_cell(grid)
        if not empty:
            return True

        row, col = empty

        for num in range(1, 10):
            if self._is_valid(grid, row, col, num):
                grid[row][col] = num

                if self._solve_recursive(grid):
                    return True

                grid[row][col] = 0

        return False

    def _is_valid(self, grid: Grid, row: int, col: int, num: int) -> bool:
        """
        Check if placing num at (row, col) is valid.

        Args:
            grid: Current grid state
            row: Row index
            col: Column index
            num: Number to place (1-9)

        Returns:
            True if placement is valid, False otherwise
        """
        # Check row
        if num in grid[row]:
            return False

        # Check column
        for r in range(9):
            if grid[r][col] == num:
                return False

        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3

        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if grid[r][c] == num:
                    return False

        return True

    def _find_empty_cell(self, grid: Grid) -> Optional[Tuple[int, int]]:
        """
        Find the next empty cell (contains 0).

        Args:
            grid: Current grid state

        Returns:
            Tuple of (row, col) if empty cell found, None otherwise
        """
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    return (r, c)
        return None

    def count_solutions(self, grid: Grid, max_count: int = 2) -> int:
        """
        Count number of solutions (up to max_count).

        Args:
            grid: 9x9 grid to solve
            max_count: Stop counting after finding this many solutions

        Returns:
            Number of solutions found
        """
        self.solutions_count = 0
        grid_copy = copy.deepcopy(grid)
        if not self._is_consistent_grid(grid_copy):
            return 0
        self._count_solutions_recursive(grid_copy, max_count)
        return self.solutions_count

    def _count_solutions_recursive(self, grid: Grid, max_count: int) -> None:
        """Recursively count solutions, stopping at max_count."""
        if self.solutions_count >= max_count:
            return

        empty = self._find_empty_cell(grid)
        if not empty:
            self.solutions_count += 1
            return

        row, col = empty

        for num in range(1, 10):
            if self._is_valid(grid, row, col, num):
                grid[row][col] = num
                self._count_solutions_recursive(grid, max_count)
                grid[row][col] = 0

    def _is_consistent_grid(self, grid: Grid) -> bool:
        """Check existing non-zero givens are mutually consistent."""
        for r in range(9):
            for c in range(9):
                num = grid[r][c]
                if num == 0:
                    continue
                grid[r][c] = 0
                valid = self._is_valid(grid, r, c, num)
                grid[r][c] = num
                if not valid:
                    return False
        return True


def solve(grid: Grid) -> Optional[Grid]:
    """Convenience function to solve a Sudoku grid."""
    solver = SudokuSolver()
    return solver.solve(grid)


def is_valid_grid(grid: Grid) -> bool:
    """
    Validate that a grid has correct structure and initial values.

    Args:
        grid: 9x9 grid to validate

    Returns:
        True if grid is valid, False otherwise
    """
    if not isinstance(grid, list) or len(grid) != 9:
        return False

    for row in grid:
        if not isinstance(row, list) or len(row) != 9:
            return False
        for cell in row:
            if not isinstance(cell, int) or cell < 0 or cell > 9:
                return False

    # Check no duplicate values in rows, cols, boxes
    solver = SudokuSolver()
    return solver._is_consistent_grid(grid)
