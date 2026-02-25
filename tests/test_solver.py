"""Tests for Sudoku solver."""

import pytest
from backend.solver.backtracking import SudokuSolver, solve, is_valid_grid


class TestSudokuSolver:
    """Tests for the backtracking solver."""

    def test_solve_easy_puzzle(self):
        """Test solving an easy Sudoku puzzle."""
        grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

        solver = SudokuSolver()
        result = solver.solve(grid)

        assert result is not None
        assert len(result) == 9
        assert all(len(row) == 9 for row in result)
        # Check no zeros in solution
        assert all(all(cell != 0 for cell in row) for row in result)

    def test_solve_convenience_function(self):
        """Test the convenience solve function."""
        grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

        result = solve(grid)
        assert result is not None

    def test_unsolvable_puzzle(self):
        """Test that an invalid puzzle returns None."""
        grid = [
            [5, 5, 0, 0, 7, 0, 0, 0, 0],  # Duplicate 5 in row
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

        solver = SudokuSolver()
        result = solver.solve(grid)
        assert result is None

    def test_already_solved_puzzle(self):
        """Test that a solved puzzle is returned as-is."""
        grid = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ]

        solver = SudokuSolver()
        result = solver.solve(grid)
        assert result == grid

    def test_count_solutions(self):
        """Test counting solutions."""
        grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

        solver = SudokuSolver()
        count = solver.count_solutions(grid)
        assert count == 1

    def test_count_solutions_returns_zero_for_invalid_givens(self):
        """Invalid givens should not report any solution."""
        grid = [
            [5, 5, 0, 0, 7, 0, 0, 0, 0],  # Duplicate 5 in row
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

        solver = SudokuSolver()
        assert solver.count_solutions(grid) == 0

    def test_is_valid_grid_valid(self):
        """Test validation of valid grid."""
        grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ]

        assert is_valid_grid(grid) is True

    def test_is_valid_grid_invalid_size(self):
        """Test validation rejects wrong size grids."""
        assert is_valid_grid([[1, 2, 3]]) is False
        assert is_valid_grid([]) is False

    def test_is_valid_grid_invalid_values(self):
        """Test validation rejects invalid cell values."""
        grid = [
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 10, 0, 7, 9],  # 10 is invalid
        ]

        assert is_valid_grid(grid) is False

    def test_empty_grid(self):
        """Test solving an empty grid (should find a solution)."""
        grid = [[0] * 9 for _ in range(9)]

        solver = SudokuSolver()
        result = solver.solve(grid)

        # Empty grid should have many solutions but should find one
        assert result is not None
        assert all(all(cell != 0 for cell in row) for row in result)
