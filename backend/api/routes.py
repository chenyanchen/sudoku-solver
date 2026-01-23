"""API routes for the Sudoku solver application."""

import base64
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from ..models.schemas import (
    SolveRequest,
    SolveResponse,
    ImageSolveResponse,
    HealthResponse,
)
from ..solver.backtracking import SudokuSolver, is_valid_grid
from ..cv.grid_detector import find_grid
from ..cv.cell_extractor import extract_cells
from ..ocr.digit_reader import DigitReader

router = APIRouter()


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def detect_grid_image(image: np.ndarray) -> Optional[np.ndarray]:
    """Detect the Sudoku grid, retrying with a simple contrast enhancement."""
    grid_img = find_grid(image)
    if grid_img is not None:
        return grid_img

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return find_grid(enhanced_bgr)


def prepare_grid_for_ocr(grid_img: np.ndarray) -> np.ndarray:
    """Enhance grid image for OCR and return a BGR image for cell extraction."""
    gray = (
        cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        if len(grid_img.shape) == 3
        else grid_img
    )
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
    enhanced = clahe.apply(gray)

    if np.mean(enhanced) <= 128:
        enhanced = cv2.bitwise_not(enhanced)

    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        import pytesseract

        tesseract_available = True
        # Try to get Tesseract version
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            tesseract_available = False
    except ImportError:
        tesseract_available = False

    return HealthResponse(status="healthy", tesseract_available=tesseract_available)


@router.post("/api/v1/sudoku:solve", response_model=SolveResponse, tags=["Sudoku"])
async def solve_sudoku(request: SolveRequest):
    """
    Solve a Sudoku puzzle from a JSON grid.

    Expected JSON format:
    {
        "grid": {
            "cells": [[row1], [row2], ...]
        }
    }
    Where each row is a list of 9 integers (0 for empty).
    """
    try:
        grid = request.grid.cells

        # Validate grid format
        if not is_valid_grid(grid):
            return SolveResponse(
                success=False,
                original=grid,
                solved=None,
                message="Invalid Sudoku grid format",
            )

        # Solve the puzzle
        solver = SudokuSolver()
        solved = solver.solve(grid)

        if solved is None:
            # Check if it's unsolvable or has multiple solutions
            solution_count = solver.count_solutions(grid, max_count=2)
            if solution_count == 0:
                message = "Puzzle has no solution"
            else:
                message = "Puzzle has multiple solutions"
            return SolveResponse(
                success=False, original=grid, solved=None, message=message
            )

        return SolveResponse(
            success=True,
            original=grid,
            solved=solved,
            message="Puzzle solved successfully",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/api/v1/sudoku:solveImage",
    response_model=ImageSolveResponse,
    tags=["Sudoku"],
)
async def solve_sudoku_from_image(
    image: UploadFile = File(...), ocr_threshold: Optional[float] = 50.0
):
    """
    Solve a Sudoku puzzle from an uploaded image.

    The image should contain a Sudoku grid. The API will:
    1. Detect the grid
    2. Extract cells
    3. Recognize digits using OCR
    4. Solve the puzzle

    Returns the original and solved grids, plus the detected grid image.
    """
    try:
        # Read uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return ImageSolveResponse(
                success=False,
                message="Failed to decode image. Please upload a valid image file.",
                original_grid=None,
                solved_grid=None,
                detected_image=None,
                confidence=None,
            )

        # Detect and extract grid from original image first
        grid_img = detect_grid_image(img)

        if grid_img is None:
            return ImageSolveResponse(
                success=False,
                message="Could not detect a Sudoku grid in the image. Please ensure the grid is clearly visible.",
                original_grid=None,
                solved_grid=None,
                detected_image=None,
                confidence=None,
            )

        # Enhance grid for OCR and extract cells
        ocr_grid = prepare_grid_for_ocr(grid_img)
        cells = extract_cells(ocr_grid)

        # Recognize digits using OCR
        reader = DigitReader()
        # Use lower threshold for faint digits
        threshold = ocr_threshold if ocr_threshold is not None else 25.0
        grid = reader.recognize_grid(cells, threshold=threshold)

        # Count recognized cells
        given_cells = sum(1 for row in grid for cell in row if cell != 0)

        if given_cells < 17:
            return ImageSolveResponse(
                success=False,
                message=f"Only {given_cells} cells were recognized. A valid Sudoku requires at least 17 given cells. Please try with a clearer image.",
                original_grid=grid,
                solved_grid=None,
                detected_image=image_to_base64(grid_img),
                confidence=None,
            )

        # Solve the puzzle
        solver = SudokuSolver()
        solved = solver.solve(grid)

        if solved is None:
            return ImageSolveResponse(
                success=False,
                message="Could not solve the detected puzzle. It may be invalid or have multiple solutions.",
                original_grid=grid,
                solved_grid=None,
                detected_image=image_to_base64(grid_img),
                confidence=None,
            )

        return ImageSolveResponse(
            success=True,
            message="Puzzle solved successfully",
            original_grid=grid,
            solved_grid=solved,
            detected_image=image_to_base64(grid_img),
            confidence=None,
        )

    except Exception as e:
        return ImageSolveResponse(
            success=False,
            message=f"Error processing image: {str(e)}",
            original_grid=None,
            solved_grid=None,
            detected_image=None,
            confidence=None,
        )


@router.post("/api/v1/sudoku:detectGrid", tags=["Sudoku"])
async def detect_grid(image: UploadFile = File(...)):
    """
    Detect and extract the Sudoku grid from an image.

    Returns the detected grid image as base64.
    """
    try:
        # Read uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # Detect grid
        grid_img = detect_grid_image(img)

        if grid_img is None:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Could not detect a Sudoku grid in the image",
                },
            )

        return {
            "success": True,
            "message": "Grid detected successfully",
            "detected_image": image_to_base64(grid_img),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
