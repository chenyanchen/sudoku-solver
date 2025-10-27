package sudoku

import (
	"fmt"
	"testing"
)

func TestBacktrackingSolve(t *testing.T) {
	type args struct {
		board [][]int
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "case 1",
			args: args{
				board: [][]int{
					{5, 3, 0, 0, 7, 0, 0, 0, 0},
					{6, 0, 0, 1, 9, 5, 0, 0, 0},
					{0, 9, 8, 0, 0, 0, 0, 6, 0},
					{8, 0, 0, 0, 6, 0, 0, 0, 3},
					{4, 0, 0, 8, 0, 3, 0, 0, 1},
					{7, 0, 0, 0, 2, 0, 0, 0, 6},
					{0, 6, 0, 0, 0, 0, 2, 8, 0},
					{0, 0, 0, 4, 1, 9, 0, 0, 5},
					{0, 0, 0, 0, 8, 0, 0, 7, 9},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			BacktrackingSolve(tt.args.board)
			if !Validate(tt.args.board) {
				t.Error("solveSudoku():")
				printBoard(tt.args.board)
			}
		})
	}
}

func printBoard(board [][]int) {
	for _, row := range board {
		for _, cell := range row {
			fmt.Print(cell, " ")
		}
		fmt.Println()
	}
}
