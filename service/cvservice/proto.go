package cvservice

import (
	"context"
	"mime/multipart"
)

// CVService represent the Computer Vision Service.
type CVService interface {
	// Recognize the image to Sudoku puzzle
	Recognize(ctx context.Context, req *RecognizeRequest) (*RecognizeResponse, error)
}

type RecognizeRequest struct {
	File *multipart.FileHeader
}

type RecognizeResponse struct {
	Puzzle       [][]int      `json:"puzzle"` // the 9*9 puzzle
	GridLocation GridLocation `json:"grid_location"`
}

type GridLocation struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}
