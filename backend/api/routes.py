"""API routes for the Sudoku solver application."""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

from ..cv.corner_cnn_detector import CornerCnnDetector
from ..cv.detector import warp_grid
from ..cv.grid_detector import has_regular_grid_lines
from ..cv.opencv_detector import OpenCVGridDetector
from ..models.schemas import (
    HealthResponse,
    RecognizeResponse,
    SolveRequest,
    SolveResponse,
)
from ..ocr.cnn_digit_reader import CnnDigitReader
from ..ocr.pipeline import SOLVED, Recognition, recognize
from ..solver.backtracking import SudokuSolver, is_valid_grid

router = APIRouter()
_CNN_READER: CnnDigitReader | None = None
_OPENCV = OpenCVGridDetector()
_CORNER_CNN: CornerCnnDetector | None = None
_LOGGER = logging.getLogger(__name__)


def image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode(".png", image)
    return base64.b64encode(buffer).decode("utf-8")


def _get_corner_cnn() -> CornerCnnDetector | None:
    global _CORNER_CNN
    if _CORNER_CNN is None:
        _CORNER_CNN = CornerCnnDetector(model_path=os.getenv("CORNER_CNN_MODEL_PATH"))
        if not _CORNER_CNN.is_ready:
            _LOGGER.info("Corner CNN unavailable: %s", _CORNER_CNN.load_error)
    return _CORNER_CNN if _CORNER_CNN.is_ready else None


@dataclass
class GridReading:
    recognition: Recognition
    warped: np.ndarray  # 450x450 grid
    detector_name: str


def read_image(image: np.ndarray) -> GridReading | None:
    """Detect, warp and recognize the grid; None if no grid is found.

    OpenCV and Corner CNN each solve images the other cannot (glare over the
    lines, a plastic sleeve; grids Corner CNN mislocates), so Corner CNN's
    corners are tried whenever OpenCV's grid does not solve. A unique solution
    is the acceptance test, so Corner CNN skips the grid-regularity check,
    which rejects good corners on photos with moiré. That check still decides
    whether an unsolved Corner CNN quad counts as a grid at all.
    """
    reader = _get_cnn_reader()

    def read(corners: np.ndarray, name: str) -> GridReading:
        warped = warp_grid(image, corners)
        return GridReading(recognize(reader, warped), warped, name)

    found = None
    detection = _OPENCV.detect(image)
    if detection is not None:
        found = read(detection.corners, detection.detector_name)
        if found.recognition.status == SOLVED:
            return found

    corner_cnn = _get_corner_cnn()
    corners = corner_cnn.predict(image) if corner_cnn else None
    if corners is not None:
        second = read(corners, "corner_cnn")
        if second.recognition.status == SOLVED:
            return second
        if found is None and has_regular_grid_lines(cv2.cvtColor(second.warped, cv2.COLOR_BGR2GRAY)):
            found = second
    return found


def _get_cnn_reader() -> CnnDigitReader:
    """Load the CNN reader once; CNN_* env vars override its thresholds."""
    global _CNN_READER
    if _CNN_READER is None:
        overrides: dict[str, float | int] = {
            arg: float(os.environ[env])
            for arg, env in (
                ("blank_threshold", "CNN_BLANK_THRESHOLD"),
                ("digit_threshold", "CNN_DIGIT_THRESHOLD"),
            )
            if env in os.environ
        }
        if "CNN_TOPK_CANDIDATES" in os.environ:
            overrides["top_k_candidates"] = int(os.environ["CNN_TOPK_CANDIDATES"])
        _CNN_READER = CnnDigitReader(os.environ["CNN_MODEL_PATH"], **overrides)
    return _CNN_READER


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        cnn_model_loaded=True,
        cnn_model_version=_get_cnn_reader().model_version,
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
    grid = request.grid.cells
    if not is_valid_grid(grid):
        return SolveResponse(
            success=False, original=grid, solved=None, message="Invalid Sudoku grid format",
        )

    solved, solution_count = SudokuSolver().solve_unique(grid)
    if solved is None:
        message = (
            "Puzzle has no solution" if solution_count == 0 else "Puzzle has multiple solutions"
        )
        return SolveResponse(success=False, original=grid, solved=None, message=message)

    return SolveResponse(
        success=True, original=grid, solved=solved, message="Puzzle solved successfully",
    )


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
    contents = await image.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return RecognizeResponse(
            success=False,
            message="Failed to decode image. Please upload a valid image file.",
        )

    reading = read_image(img)
    if reading is None:
        return RecognizeResponse(
            success=False,
            message="Could not detect a Sudoku grid in the image.",
        )
    result = reading.recognition

    message = f"Recognized {result.givens} givens"
    if result.status != SOLVED:
        message += f" ({result.status}: some digits may be misread)"
    return RecognizeResponse(
        success=True,
        message=message,
        grid=result.grid,
        detected_image=image_to_base64(reading.warped),
        detector_name=reading.detector_name,
        ocr_confidence=result.metadata["average_confidence"],
        ocr_engine="cnn",
        ocr_model_version=result.metadata["model_version"],
        ocr_latency_ms=sum(result.timing.values()),
    )
