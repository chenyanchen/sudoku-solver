package main

import (
	"io"
	"net/http"
	"os"

	"github.com/chenyanchen/sudoku-solver/service/cvservice"
	"github.com/gin-gonic/gin"
	"github.com/rs/zerolog/log"
)

func main() {
	e := gin.Default()
	v1 := e.Group("/api").
		Group("/v1")

	client, err := cvservice.NewClient("http://localhost:8000", nil)
	if err != nil {
		log.Fatal().Err(err).Msg("create CV service client")
	}
	sudokuHandler := NewSudokuSolveHandler(client)
	v1.POST("/solve", sudokuHandler.Solve)

	if err = e.Run(":8080"); err != nil {
		log.Fatal().Err(err).Msg("run server")
	}
}

type SudokuSolveHandler struct {
	cvService cvservice.CVService // client of recognize service
}

func NewSudokuSolveHandler(cvService cvservice.CVService) *SudokuSolveHandler {
	return &SudokuSolveHandler{cvService: cvService}
}

func (s *SudokuSolveHandler) Solve(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		log.Err(err).Msg("read file from form")
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read form file", "message": err.Error()})
		return
	}

	tmp, err := os.CreateTemp("", "sudoku_*")
	if err != nil {
		log.Err(err).Msg("create temporary file")
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to create temporary file", "message": err.Error()})
		return
	}
	defer tmp.Close()

	f, err := file.Open()
	if err != nil {
		log.Err(err).Msg("open file")
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to open form file", "message": err.Error()})
		return
	}
	defer f.Close()

	_, err = io.Copy(tmp, f)

	resp, err := s.cvService.Recognize(c, &cvservice.RecognizeRequest{File: file})
	if err != nil {
		log.Err(err).Msg("recognize image")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to recognize", "message": err.Error()})
		return
	}

	log.Info().Any("recognize", resp)
	c.JSON(http.StatusOK, gin.H{"message": "todo"})
}
