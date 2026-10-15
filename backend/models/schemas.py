"""Pydantic models for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SudokuGrid(BaseModel):
    """A Sudoku grid."""

    cells: list[list[int]] = Field(description="9x9 grid (0 for empty cells)")

    class Config:
        json_schema_extra = {
            "example": [
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
        }


class SolveRequest(BaseModel):
    """Request to solve a Sudoku grid."""

    grid: SudokuGrid = Field(description="The Sudoku puzzle to solve")


class SolveResponse(BaseModel):
    """Response from solving a Sudoku."""

    success: bool = Field(description="Whether the puzzle was solved")
    original: list[list[int]] = Field(description="Original grid")
    solved: list[list[int]] | None = Field(description="Solved grid (if successful)")
    message: str = Field(description="Status message")
    status: str = Field(description="solved, no_solution or invalid")


class RecognizeResponse(BaseModel):
    """Response from recognizing a Sudoku grid in an image."""

    success: bool = Field(description="Whether recognition was successful")
    message: str = Field(description="Status message")
    grid: list[list[int]] | None = Field(
        default=None, description="9x9 recognized grid (0 for empty cells)"
    )
    detected_image: str | None = Field(
        default=None, description="Base64 encoded detected grid image"
    )
    detector_name: str | None = Field(
        default=None, description="Grid detector that found the puzzle (opencv or corner_cnn)"
    )
    ocr_confidence: float | None = Field(
        default=None, description="Average OCR confidence"
    )
    ocr_engine: str | None = Field(
        default=None, description="OCR engine used for recognition"
    )
    ocr_model_version: str | None = Field(
        default=None, description="OCR model version"
    )
    ocr_latency_ms: float | None = Field(
        default=None, description="OCR stage latency in milliseconds"
    )
    unique: bool = Field(
        default=False, description="Whether the givens have exactly one solution: evidence they were read right"
    )
    status: str | None = Field(
        default=None, description="unique, ambiguous (several solutions) or unsolvable"
    )
    token: str | None = Field(
        default=None, description="Authorizes a correction of this reading via /api/v1/sudoku:feedback"
    )


class FeedbackRequest(BaseModel):
    """The uploader's corrected givens for a reading."""

    token: str = Field(description="Token returned by /api/v1/sudoku:recognize")
    grid: SudokuGrid = Field(description="Corrected givens")


class StatsResponse(BaseModel):
    """Public solve rate over distinct uploaded images."""

    solved: int
    total: int
    percentage: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    cnn_model_loaded: bool | None = Field(
        default=None, description="Whether CNN OCR model is loaded and ready"
    )
    cnn_model_version: str | None = Field(
        default=None, description="Loaded CNN OCR model version"
    )
