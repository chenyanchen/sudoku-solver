"""Tests for Sudoku solver."""

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


def test_solve_unique_rejects_ambiguous_puzzle():
    empty = [[0] * 9 for _ in range(9)]
    solution, count = SudokuSolver().solve_unique(empty)
    assert solution is None and count == 2


# An uploaded puzzle with two solutions: 1 and 2 swap in r8-9 c6/c8.
AMBIGUOUS = [
    [0, 0, 0, 5, 0, 0, 0, 4, 3], [4, 0, 0, 1, 6, 8, 7, 0, 9], [5, 9, 0, 0, 0, 4, 0, 8, 6],
    [8, 3, 4, 0, 1, 0, 5, 0, 2], [0, 0, 1, 2, 0, 5, 6, 3, 0], [0, 0, 0, 9, 8, 0, 0, 7, 0],
    [2, 1, 0, 8, 7, 0, 3, 0, 0], [0, 6, 9, 4, 0, 0, 8, 0, 0], [7, 0, 0, 6, 3, 0, 0, 0, 5],
]


def test_solve_endpoint_returns_a_solution_however_many_there_are():
    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as client:
        for grid in (AMBIGUOUS, [[0] * 9 for _ in range(9)]):
            body = client.post("/api/v1/sudoku:solve", json={"grid": {"cells": grid}}).json()
            assert body["success"] and body["status"] == "solved"
            solved = body["solved"]
            assert all(v for row in solved for v in row) and is_valid_grid(solved)
            assert all(solved[r][c] == grid[r][c] for r in range(9) for c in range(9) if grid[r][c])


# A 16x16 upload (values 10-16 are the letters A-G), with one solution.
PUZZLE_16 = [
    [0, 0, 0, 15, 12, 13, 16, 7, 0, 0, 4, 0, 0, 6, 1, 0], [0, 0, 3, 9, 10, 0, 1, 8, 7, 15, 0, 0, 16, 4, 0, 0],
    [13, 0, 1, 4, 9, 15, 0, 0, 0, 0, 12, 5, 8, 0, 2, 0], [10, 0, 0, 16, 0, 4, 2, 0, 13, 0, 8, 0, 7, 0, 3, 9],
    [11, 6, 0, 0, 0, 0, 0, 0, 15, 3, 14, 13, 0, 12, 8, 16], [0, 0, 0, 13, 15, 12, 0, 0, 1, 0, 0, 8, 0, 0, 6, 0],
    [7, 0, 12, 0, 0, 0, 6, 0, 0, 0, 2, 10, 14, 3, 5, 0], [0, 9, 0, 0, 5, 8, 11, 3, 12, 6, 7, 0, 2, 10, 13, 1],
    [0, 0, 8, 3, 2, 6, 4, 0, 0, 0, 16, 9, 1, 0, 10, 7], [14, 0, 4, 0, 8, 3, 0, 0, 2, 7, 0, 11, 13, 16, 0, 0],
    [5, 16, 0, 6, 0, 7, 0, 11, 8, 0, 0, 15, 0, 2, 0, 0], [1, 2, 0, 7, 0, 5, 0, 0, 0, 10, 0, 0, 11, 0, 9, 12],
    [0, 0, 2, 8, 4, 0, 12, 15, 0, 0, 0, 0, 10, 14, 0, 0], [9, 0, 0, 0, 6, 0, 0, 0, 16, 0, 3, 0, 0, 13, 0, 8],
    [3, 0, 0, 5, 7, 0, 0, 9, 6, 0, 0, 0, 15, 0, 11, 0], [0, 0, 0, 11, 0, 0, 0, 10, 4, 0, 0, 2, 6, 0, 0, 5],
]


def test_solves_a_16x16_puzzle_uniquely():
    solution, count = SudokuSolver().solve_unique(PUZZLE_16)
    assert count == 1 and is_valid_grid(solution)
    assert sorted(solution[0]) == list(range(1, 17))
    assert all(solution[r][c] == PUZZLE_16[r][c] for r in range(16) for c in range(16) if PUZZLE_16[r][c])


def test_16x16_boxes_are_4x4():
    grid = [[0] * 16 for _ in range(16)]
    grid[0][0] = grid[3][3] = 16  # same 4x4 box, different row and column
    assert not is_valid_grid(grid)
    grid[3][3], grid[4][4] = 0, 16  # next box
    assert is_valid_grid(grid)
