package sudoku

func BacktrackingSolve(board [][]int) {
	backtrackingSolve(board, 0, 0)
}

func backtrackingSolve(board [][]int, r, c int) bool {
	r, c, solved := nextEmptyCell(board, r, c)
	if solved {
		return true
	}

	for i := 1; i <= 9; i++ {
		if !isValid(board, r, c, i) {
			continue
		}
		board[r][c] = i
		if backtrackingSolve(board, r, c) {
			return true
		}
		board[r][c] = 0 // backtracking
	}

	return false
}

func nextEmptyCell(board [][]int, row, col int) (r, c int, solved bool) {
	for ; row < 9; row++ {
		for ; col < 9; col++ {
			if board[row][col] == 0 {
				return row, col, false
			}
		}
		col = 0
	}
	return 0, 0, true
}

func isValid(board [][]int, row, col int, digit int) bool {
	for i := 0; i < 9; i++ {
		if board[row][i] == digit ||
			board[i][col] == digit ||
			board[row/3*3+i/3][col/3*3+i%3] == digit {
			return false
		}
	}
	return true
}
