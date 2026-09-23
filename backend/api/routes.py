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

from ..cv.cell_extractor import extract_cells
from ..cv.detector import ChainDetector, warp_grid
from ..cv.opencv_detector import OpenCVGridDetector
from ..cv.corner_cnn_detector import CornerCnnDetector
from ..models.schemas import (
    HealthResponse,
    RecognizeResponse,
    SolveRequest,
    SolveResponse,
)
from ..ocr.cnn_digit_reader import CnnDigitReader
from ..solver.backtracking import SudokuSolver, is_valid_grid, is_valid_placement

router = APIRouter()
_CNN_READER: CnnDigitReader | None = None
_DETECTOR: ChainDetector | None = None
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


def _get_detector() -> ChainDetector:
    """Lazy-init the detector chain: CornerCNN (preferred) → OpenCV (fallback)."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    detectors = []

    corner_cnn = CornerCnnDetector(
        model_path=os.getenv("CORNER_CNN_MODEL_PATH"),
    )
    if corner_cnn.is_ready:
        detectors.append(corner_cnn)
        _LOGGER.info("Corner CNN grid detector ready")
    else:
        _LOGGER.info("Corner CNN unavailable: %s", corner_cnn.load_error)

    detectors.append(OpenCVGridDetector())
    _LOGGER.info("OpenCV grid detector ready (fallback)")

    _DETECTOR = ChainDetector(detectors)
    return _DETECTOR


def detect_grid_image(image: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Detect grid and warp to 450x450. Returns (warped, detector_name) or None."""
    result = _get_detector().detect(image)
    if result is None:
        return None
    return warp_grid(image, result.corners), result.detector_name


def prepare_grid_for_ocr(grid_img: np.ndarray) -> np.ndarray:
    """Enhance grid image for OCR and return a BGR image for cell extraction."""
    gray = (
        cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
        if len(grid_img.shape) == 3
        else grid_img
    )
    if np.mean(gray) <= 128:
        gray = cv2.bitwise_not(gray)
    # Threshold against local lighting before cell-level Otsu can amplify shadows.
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


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


def recognize_with_border_retry(
    reader: CnnDigitReader, ocr_grid: np.ndarray, cells: list[np.ndarray]
) -> tuple[list[list[int]], dict]:
    """Retry weak OCR on the cell centers, then accept only an unambiguous puzzle."""
    grid, metadata = reader.recognize_grid_with_metadata(cells)
    if sum(value != 0 for row in grid for value in row) >= 17:
        return grid, metadata

    # Thick screen grid lines survive the normal 1px cell border crop.
    centered = [
        ocr_grid[
            r * 50 + 10 : (r + 1) * 50 - 10,
            c * 50 + 10 : (c + 1) * 50 - 10,
        ]
        for r in range(9) for c in range(9)
    ]
    retry, retry_meta = reader.recognize_grid_with_metadata(centered, threshold=0.3)
    retry_givens = sum(value != 0 for row in retry for value in row)
    if retry_givens < 17 or retry_givens <= sum(
        value != 0 for row in grid for value in row
    ):
        return grid, metadata

    solver = SudokuSolver()
    solutions = solver.count_solutions(retry, max_count=2)
    if solutions == 1:
        return retry, retry_meta
    if solutions != 0:
        return grid, metadata

    # One uncertain glyph may be wrong even when all other cells are legible.
    corrections = []
    for pred in retry_meta["cell_predictions"]:
        row, col = pred["row"], pred["col"]
        old = retry[row][col]
        if old == 0 or pred["confidence"] >= 0.8:
            continue
        for digit, prob in pred["candidates"]:
            if (digit == old or prob < 0.15
                    or not is_valid_placement(retry, row, col, digit)):
                continue
            retry[row][col] = digit
            if solver.count_solutions(retry, max_count=2) == 1:
                corrections.append((row, col, digit, prob))
            retry[row][col] = old
            if len(corrections) > 1:
                return grid, metadata
    if len(corrections) != 1:
        return grid, metadata
    row, col, digit, prob = corrections[0]
    retry[row][col] = digit
    prediction = retry_meta["cell_predictions"][row * 9 + col]
    prediction["value"], prediction["confidence"] = digit, prob
    return retry, retry_meta


@router.post(
    "/api/v1/sudoku:recognize",
    response_model=RecognizeResponse,
    tags=["Sudoku"],
)
async def recognize_sudoku(image: UploadFile = File(...)):
    """Recognize a Sudoku grid from an uploaded image.

    Detects the grid, extracts cells, and runs OCR.
    Returns the recognized grid — call `/api/v1/sudoku:solve` to solve it.
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return RecognizeResponse(
                success=False,
                message="Failed to decode image. Please upload a valid image file.",
            )

        det_result = detect_grid_image(img)
        if det_result is None:
            return RecognizeResponse(
                success=False,
                message="Could not detect a Sudoku grid in the image.",
            )

        grid_img, detector_name = det_result
        ocr_grid = prepare_grid_for_ocr(grid_img)
        cells = extract_cells(ocr_grid)
        detected_b64 = image_to_base64(grid_img)

        reader, engine, reader_error = _resolve_ocr_reader()
        if reader is None:
            return RecognizeResponse(
                success=False,
                message=f"OCR engine '{engine}' is not ready: {reader_error}",
                detected_image=detected_b64,
                detector_name=detector_name,
                ocr_engine=engine,
            )

        ocr_start = time.perf_counter()
        grid, metadata = recognize_with_border_retry(reader, ocr_grid, cells)
        ocr_latency_ms = (time.perf_counter() - ocr_start) * 1000.0

        given_cells = sum(1 for row in grid for cell in row if cell != 0)

        return RecognizeResponse(
            success=True,
            message=f"Recognized {given_cells} givens",
            grid=grid,
            detected_image=detected_b64,
            detector_name=detector_name,
            ocr_confidence=metadata.get("average_confidence"),
            ocr_engine=engine,
            ocr_model_version=metadata.get("model_version"),
            ocr_latency_ms=ocr_latency_ms,
        )

    except Exception as e:
        return RecognizeResponse(
            success=False,
            message=f"Error processing image: {e}",
        )
