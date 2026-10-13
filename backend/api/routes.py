"""API routes for the Sudoku solver application."""

from __future__ import annotations

import base64
import logging
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile

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
from ..ocr.pipeline import SOLVED, recognize
from ..solver.backtracking import SudokuSolver, is_valid_grid

router = APIRouter()
_CNN_READER: CnnDigitReader | None = None
_DETECTOR: ChainDetector | None = None
_LOGGER = logging.getLogger(__name__)

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

    det_result = detect_grid_image(img)
    if det_result is None:
        return RecognizeResponse(
            success=False,
            message="Could not detect a Sudoku grid in the image.",
        )
    grid_img, detector_name = det_result

    ocr_start = time.perf_counter()
    result = recognize(_get_cnn_reader(), grid_img)
    ocr_latency_ms = (time.perf_counter() - ocr_start) * 1000.0

    message = f"Recognized {result.givens} givens"
    if result.status != SOLVED:
        message += f" ({result.status}: some digits may be misread)"
    return RecognizeResponse(
        success=True,
        message=message,
        grid=result.grid,
        detected_image=image_to_base64(grid_img),
        detector_name=detector_name,
        ocr_confidence=result.metadata["average_confidence"],
        ocr_engine="cnn",
        ocr_model_version=result.metadata["model_version"],
        ocr_latency_ms=ocr_latency_ms,
    )
