"""API routes for the Sudoku solver application."""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import TypeVar

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..cv.cell_extractor import extract_cells
from ..cv.grid_detector import find_grid
from ..models.schemas import (
    HealthResponse,
    ImageSolveResponse,
    SolveRequest,
    SolveResponse,
)
from ..ocr.cnn_digit_reader import CnnDigitReader
from ..ocr.grid_repair import try_repair_grid_with_candidates
from ..solver.backtracking import SudokuSolver, is_valid_grid

router = APIRouter()
_CNN_READER: CnnDigitReader | None = None
_LOGGER = logging.getLogger(__name__)

_T = TypeVar("_T", int, float)


def _env(name: str, default: _T) -> _T:
    """Read an environment variable, converting to the same type as *default*."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return type(default)(raw)
    except (TypeError, ValueError):
        return default


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def detect_grid_image(image: np.ndarray) -> np.ndarray | None:
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


def _ocr_engine() -> str:
    return os.getenv("OCR_ENGINE", "cnn").strip().lower()


def _get_cnn_reader() -> tuple[CnnDigitReader | None, str | None]:
    global _CNN_READER

    if _CNN_READER is not None:
        return _CNN_READER, None

    reader = CnnDigitReader(
        model_path=os.getenv("CNN_MODEL_PATH"),
        blank_threshold=_env("CNN_BLANK_THRESHOLD", 0.65),
        digit_threshold=_env("CNN_DIGIT_THRESHOLD", 0.55),
        rerank_confidence=_env("CNN_RERANK_CONFIDENCE", 0.80),
        top_k_candidates=_env("CNN_TOPK_CANDIDATES", 4),
        strict=False,
    )

    if reader.is_ready:
        _CNN_READER = reader
        return _CNN_READER, None

    return None, reader.load_error or "CNN OCR reader initialization failed"


def _resolve_ocr_reader() -> tuple[CnnDigitReader | None, str, str | None]:
    engine = _ocr_engine()
    if engine != "cnn":
        _LOGGER.warning("Unsupported OCR_ENGINE=%s, fallback to cnn", engine)
    reader, err = _get_cnn_reader()
    return reader, "cnn", err


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cnn_reader, _ = _get_cnn_reader()

    return HealthResponse(
        status="healthy",
        cnn_model_loaded=cnn_reader is not None,
        cnn_model_version=cnn_reader.model_version if cnn_reader else None,
    )


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


def _image_response(
    message: str,
    *,
    success: bool = False,
    original_grid: list[list[int]] | None = None,
    solved_grid: list[list[int]] | None = None,
    detected_image: str | None = None,
    confidence: float | None = None,
    ocr_engine: str | None = None,
    ocr_model_version: str | None = None,
    ocr_latency_ms: float | None = None,
) -> ImageSolveResponse:
    """Build an ImageSolveResponse with sensible defaults for optional fields."""
    return ImageSolveResponse(
        success=success,
        message=message,
        original_grid=original_grid,
        solved_grid=solved_grid,
        detected_image=detected_image,
        confidence=confidence,
        ocr_engine=ocr_engine,
        ocr_model_version=ocr_model_version,
        ocr_latency_ms=ocr_latency_ms,
    )


@router.post(
    "/api/v1/sudoku:solveImage",
    response_model=ImageSolveResponse,
    tags=["Sudoku"],
)
async def solve_sudoku_from_image(
    image: UploadFile = File(...)
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
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return _image_response(
                "Failed to decode image. Please upload a valid image file."
            )

        grid_img = detect_grid_image(img)
        if grid_img is None:
            return _image_response(
                "Could not detect a Sudoku grid in the image. "
                "Please ensure the grid is clearly visible."
            )

        ocr_grid = prepare_grid_for_ocr(grid_img)
        cells = extract_cells(ocr_grid)
        detected_b64 = image_to_base64(grid_img)

        reader, engine, reader_error = _resolve_ocr_reader()
        if reader is None:
            return _image_response(
                f"OCR engine '{engine}' is not ready: {reader_error}",
                detected_image=detected_b64,
                ocr_engine=engine,
            )

        ocr_start = time.perf_counter()
        ocr_confidence: float | None = None
        ocr_model_version: str | None = None
        cell_predictions: list[dict] | None = None

        grid, metadata = reader.recognize_grid_with_metadata(cells, threshold=None)
        ocr_confidence = metadata.get("average_confidence")
        ocr_model_version = metadata.get("model_version")
        raw_predictions = metadata.get("cell_predictions")
        if isinstance(raw_predictions, list):
            cell_predictions = raw_predictions

        ocr_latency_ms = (time.perf_counter() - ocr_start) * 1000.0

        # Shared keyword arguments for all remaining responses.
        ocr_kwargs: dict = dict(
            detected_image=detected_b64,
            confidence=ocr_confidence,
            ocr_engine=engine,
            ocr_model_version=ocr_model_version,
            ocr_latency_ms=ocr_latency_ms,
        )

        given_cells = sum(1 for row in grid for cell in row if cell != 0)
        if given_cells < 17:
            return _image_response(
                f"Only {given_cells} cells were recognized. A valid Sudoku "
                "requires at least 17 given cells. Please try with a clearer image.",
                original_grid=grid,
                **ocr_kwargs,
            )

        solver = SudokuSolver()
        solved = solver.solve(grid)
        repaired_cells = 0

        if solved is None:
            repaired_grid, repaired_solution, repair_info = (
                try_repair_grid_with_candidates(
                    grid=grid,
                    cell_predictions=cell_predictions,
                    max_changes=_env("CNN_REPAIR_MAX_CHANGES", 2),
                    max_cells=_env("CNN_REPAIR_MAX_CELLS", 14),
                )
            )
            if repaired_solution is not None:
                grid = repaired_grid
                solved = repaired_solution
                repaired_cells = int(repair_info.get("changes", 0))

        if solved is None:
            return _image_response(
                "Could not solve the detected puzzle. "
                "It may be invalid or have multiple solutions.",
                original_grid=grid,
                **ocr_kwargs,
            )

        message = "Puzzle solved successfully"
        if repaired_cells > 0:
            message += f" (OCR auto-repaired {repaired_cells} cell(s))"

        return _image_response(
            message,
            success=True,
            original_grid=grid,
            solved_grid=solved,
            **ocr_kwargs,
        )

    except Exception as e:
        return _image_response(
            f"Error processing image: {e}",
            ocr_engine=_ocr_engine(),
        )


@router.post("/api/v1/sudoku:detectGrid", tags=["Sudoku"])
async def detect_grid(image: UploadFile = File(...)):
    """Detect and extract the Sudoku grid from an image.

    Returns the detected grid image as base64.
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

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
