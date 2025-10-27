package sudoku

import "testing"

func TestValidate(t *testing.T) {
	type args struct {
		board [][]int
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "case 1",
			args: args{
				board: [][]int{
					{2, 4, 3, 1, 5, 6, 7, 9, 8},
					{1, 5, 8, 7, 3, 9, 2, 4, 6},
					{6, 7, 9, 2, 8, 4, 3, 5, 1},
					{4, 2, 6, 5, 7, 1, 8, 3, 9},
					{9, 8, 1, 3, 6, 2, 4, 7, 5},
					{5, 3, 7, 4, 9, 8, 1, 6, 2},
					{3, 1, 5, 6, 2, 7, 9, 8, 4},
					{8, 6, 4, 9, 1, 3, 5, 2, 7},
					{7, 9, 2, 8, 4, 5, 6, 1, 3},
				},
			},
			want: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Validate(tt.args.board); got != tt.want {
				t.Errorf("Validate() = %v, want %v", got, tt.want)
			}
		})
	}
}
