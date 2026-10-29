"""API routes for the Sudoku solver application."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from ..cv.corner_cnn_detector import CornerCnnDetector
from ..cv.detector import warp_grid
from ..cv.grid_detector import has_regular_grid_lines, snap_to_lattice
from ..cv.opencv_detector import OpenCVGridDetector
from ..models.schemas import (
    FeedbackRequest,
    HealthResponse,
    RecognizeResponse,
    SolveRequest,
    SolveResponse,
    StatsResponse,
)
from ..ocr.cnn_digit_reader import CnnDigitReader
from ..ocr.pipeline import AMBIGUOUS, UNIQUE, UNSOLVABLE, Recognition, recognize
from ..records import Records
from ..solver.backtracking import SudokuSolver, is_valid_grid

router = APIRouter()
_CNN_READER: CnnDigitReader | None = None
_OPENCV = OpenCVGridDetector()
_CORNER_CNN: CornerCnnDetector | None = None
_LOGGER = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Recognition is CPU bound (~50-500 ms); more concurrent requests than cores only queue up.
_RECOGNIZE_SLOTS = threading.BoundedSemaphore(int(os.getenv("SUDOKU_RECOGNIZE_SLOTS", "2")))
_RECORDS: Records | None = None


class _RateLimit:
    """At most `limit` events per key in a sliding `window` of seconds."""

    def __init__(self, limit: int, window: float = 60.0):
        self.limit, self.window = limit, window
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if len(self._events) > 10_000:  # forget idle clients
                for k in [k for k, q in self._events.items() if not q or now - q[-1] > self.window]:
                    del self._events[k]
            events = self._events[key]
            while events and now - events[0] > self.window:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(now)
            return True


_RECOGNIZE_LIMIT = _RateLimit(int(os.getenv("SUDOKU_RECOGNIZE_PER_MINUTE", "20")))


def _get_records() -> Records | None:
    """Records are kept only when SUDOKU_DATA_DIR is set (the deployed site)."""
    global _RECORDS
    if _RECORDS is None and os.getenv("SUDOKU_DATA_DIR"):
        _RECORDS = Records(Path(os.environ["SUDOKU_DATA_DIR"]))
    return _RECORDS


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
    whether an unsolved Corner CNN quad counts as a grid at all. Corner CNN's
    corners refined on a crop around its first quad are tried last: they fix
    grids that fill a small part of a photo but misplace some others. They are
    read even when the first quad solves: a quad a few percent off cuts the
    digits along one edge, and the givens left can still have one solution.
    The refined reading wins only with the same solution and more givens.
    A Corner CNN quad that does not solve is snapped to the grid lattice
    around it: the lattice extends past the image when a screenshot crops
    the grid, where Corner CNN cannot put a corner.

    Some published puzzles are ambiguous, so no reading of them is unique.
    When nothing solves uniquely, a reading with solutions beats one with
    none (certainly misread), provided it has a puzzle's worth of givens: an
    empty or partial grid has solutions too. Otherwise a warp with regular
    grid lines beats one without: OpenCV's fallback quad is often a single
    cell or 3x3 box, whose blank reading leaves the uploader nothing to fix.
    """
    reader = _get_cnn_reader()

    def read(corners: np.ndarray, name: str) -> GridReading:
        warped = warp_grid(image, corners)
        return GridReading(recognize(reader, warped), warped, name)

    def read_corner_cnn(corners: np.ndarray) -> GridReading:
        """Corner CNN's quad, or the grid lattice fitted around it if that reads better."""
        reading = read(corners, "corner_cnn")
        if reading.recognition.status == UNIQUE:
            return reading
        lattice = snap_to_lattice(image, corners, cells=9)
        if lattice is None:
            return reading
        snapped = read(lattice[1], "corner_cnn")
        if snapped.recognition.status == UNIQUE or _rank(snapped) > _rank(reading):
            return snapped
        return reading

    found = None
    detection = _OPENCV.detect(image)
    if detection is not None:
        found = read(detection.corners, detection.detector_name)
        if found.recognition.status == UNIQUE:
            return found
    best = _rank(found)

    corner_cnn = _get_corner_cnn()
    corners = corner_cnn.predict(image) if corner_cnn else None
    if corner_cnn is None or corners is None:
        return found
    solved = None
    for refined in (False, True):
        if refined:
            corners = corner_cnn.refine(image, corners)
            if corners is None:
                break
        second = read_corner_cnn(corners)
        if second.recognition.status == UNIQUE:
            if solved is None:
                solved = second
            elif (second.recognition.solution == solved.recognition.solution
                  and second.recognition.givens > solved.recognition.givens):
                return second
            continue
        rank = _rank(second)
        if rank > best:
            found, best = second, rank
    return solved or found


# No puzzle with fewer givens has a unique solution, so none is published.
MIN_PUZZLE_GIVENS = 17


def _rank(reading: GridReading | None) -> int:
    """How worth showing an unsolved reading is; an unsolved Corner CNN quad
    must outrank OpenCV's reading (or its absence) to replace it."""
    if reading is None:
        return 0
    if _may_be_exact(reading):
        return 2
    return int(has_regular_grid_lines(cv2.cvtColor(reading.warped, cv2.COLOR_BGR2GRAY)))


def _may_be_exact(reading: GridReading) -> bool:
    """A non-unique reading that could still be the puzzle as printed."""
    result = reading.recognition
    return result.status == AMBIGUOUS and result.givens >= MIN_PUZZLE_GIVENS


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
def solve_sudoku(request: SolveRequest):
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
            status="invalid",
        )

    # Any solution will do: a puzzle with several solutions is still solved by
    # each of them, and misread digits are for the uploader to correct.
    solved = SudokuSolver().solve(grid)
    if solved is None:
        return SolveResponse(
            success=False, original=grid, solved=None, message="Puzzle has no solution",
            status="no_solution",
        )

    return SolveResponse(
        success=True, original=grid, solved=solved, message="Puzzle solved successfully",
        status="solved",
    )


@router.post(
    "/api/v1/sudoku:recognize",
    response_model=RecognizeResponse,
    tags=["Sudoku"],
)
def recognize_sudoku(
    request: Request,
    image: UploadFile = File(...),
):
    """Recognize a Sudoku grid from an uploaded image.

    Detects the grid, extracts cells, and runs OCR.
    Returns the recognized grid — call `/api/v1/sudoku:solve` to solve it.
    A plain `def`: FastAPI runs it in a thread pool, so recognition does not
    block the event loop.
    """
    if not _RECOGNIZE_LIMIT.allow(request.client.host if request.client else "-"):
        raise HTTPException(429, "Too many uploads; please wait a minute.")
    contents = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return RecognizeResponse(
            success=False,
            message="Failed to decode image. Please upload a valid image file.",
        )

    if not _RECOGNIZE_SLOTS.acquire(timeout=30):
        raise HTTPException(503, "Server busy; please try again.")
    try:
        start = time.perf_counter()
        reading = read_image(img)
        latency_ms = (time.perf_counter() - start) * 1000.0
    finally:
        _RECOGNIZE_SLOTS.release()

    result = reading.recognition if reading else None
    records = _get_records()
    token = records.add(
        contents, img.shape[:2],
        detector=reading.detector_name if reading else None,
        status=result.status if result else None,
        grid=result.grid if result else None,
        latency_ms=latency_ms,
        ocr_model=result.metadata["model_version"] if result else None,
    ) if records else None

    if reading is None:
        return RecognizeResponse(
            success=False,
            message="Could not detect a Sudoku grid in the image.",
            token=token,
        )
    message = f"Recognized {result.givens} givens"
    if result.status == UNSOLVABLE:
        message += " (no solution: some digits may be misread)"
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
        unique=result.status == UNIQUE,
        status=result.status,
        token=token,
    )


@router.post("/api/v1/sudoku:feedback", tags=["Sudoku"])
def sudoku_feedback(request: FeedbackRequest):
    """Record the uploader's corrected givens for a reading (training data)."""
    grid = request.grid.cells
    if not is_valid_grid(grid):
        raise HTTPException(422, "Invalid Sudoku grid")
    records = _get_records()
    if records is None:
        return {"stored": False}
    if not records.correct(request.token, grid):
        raise HTTPException(404, "Unknown reading")
    return {"stored": True}


@router.get("/api/v1/stats", response_model=StatsResponse, tags=["Sudoku"])
def solve_stats():
    """Solved / total over distinct uploaded images, by their latest reading."""
    records = _get_records()
    return StatsResponse(**records.stats()) if records else StatsResponse(solved=0, total=0, percentage=0.0)
