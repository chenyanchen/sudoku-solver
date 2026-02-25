"""Pydantic models for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SudokuCell(BaseModel):
    """A single Sudoku cell."""

    value: int = Field(ge=0, le=9, description="Cell value (0 for empty)")
    row: int = Field(ge=0, le=8, description="Row index (0-8)")
    col: int = Field(ge=0, le=8, description="Column index (0-8)")


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


class ImageSolveResponse(BaseModel):
    """Response from solving a Sudoku from an image."""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="Status message")
    original_grid: list[list[int]] | None = Field(description="OCR-extracted grid")
    solved_grid: list[list[int]] | None = Field(description="Solved grid")
    detected_image: str | None = Field(
        description="Base64 encoded detected grid image"
    )
    confidence: float | None = Field(description="Overall OCR confidence")
    ocr_engine: str | None = Field(
        default=None, description="OCR engine used for recognition"
    )
    ocr_model_version: str | None = Field(
        default=None, description="OCR model version (for model-based engines)"
    )
    ocr_latency_ms: float | None = Field(
        default=None, description="OCR stage latency in milliseconds"
    )


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: str | None = Field(description="Detailed error information")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    tesseract_available: bool = Field(description="Whether Tesseract OCR is available")
    cnn_model_loaded: bool | None = Field(
        default=None, description="Whether CNN OCR model is loaded and ready"
    )
    cnn_model_version: str | None = Field(
        default=None, description="Loaded CNN OCR model version"
    )


class GridDetectionInfo(BaseModel):
    """Information about grid detection."""

    found: bool = Field(description="Whether a grid was found")
    corners: list[list[int]] | None = Field(description="Corner points of the grid")
    confidence: float | None = Field(description="Detection confidence")


class OCRInfo(BaseModel):
    """Information about OCR processing."""

    total_cells: int = Field(description="Total cells processed")
    recognized_cells: int = Field(description="Number of cells with recognized digits")
    empty_cells: int = Field(description="Number of empty cells")
    average_confidence: float | None = Field(
        description="Average recognition confidence"
    )
