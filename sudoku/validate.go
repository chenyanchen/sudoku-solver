package sudoku

func Validate(board [][]int) bool {
	var rows, cols, boxes [9][9]bool
	for row := 0; row < 9; row++ {
		for col := 0; col < 9; col++ {
			cell := board[row][col]
			if cell == 0 {
				return false
			}

			digit := cell - 1
			boxIndex := row/3*3 + col/3
			if rows[row][digit] || cols[col][digit] || boxes[boxIndex][digit] {
				return false
			}

			rows[row][digit], cols[col][digit], boxes[boxIndex][digit] = true, true, true
		}
	}
	return true
}
