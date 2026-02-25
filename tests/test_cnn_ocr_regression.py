"""Regression tests for CNN OCR quality on sample screenshots."""

from pathlib import Path

import cv2
import pytest

from backend.api.routes import detect_grid_image, prepare_grid_for_ocr
from backend.cv.cell_extractor import extract_cells
from backend.ocr.cnn_digit_reader import CnnDigitReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_IMAGE_ROOT = PROJECT_ROOT / "data" / "raw" / "images"
MODEL_PATH = PROJECT_ROOT / "models" / "releases" / "sudoku_digit_cnn_v1.1.onnx"

EXPECTED_SAMPLE_GRIDS = {
    "sudoku_2.png": [
        [6, 0, 8, 0, 0, 3, 0, 2, 4],
        [4, 3, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 5, 0, 0, 8, 0],
        [8, 6, 0, 4, 7, 0, 0, 3, 0],
        [0, 7, 4, 1, 6, 2, 8, 9, 5],
        [1, 0, 0, 5, 0, 0, 0, 0, 7],
        [2, 0, 6, 0, 4, 0, 1, 0, 0],
        [0, 4, 3, 8, 0, 0, 6, 0, 0],
        [0, 8, 0, 7, 2, 6, 9, 0, 0],
    ],
    "sudoku_3.png": [
        [3, 0, 2, 1, 0, 6, 9, 0, 0],
        [0, 8, 0, 0, 0, 0, 0, 0, 2],
        [0, 7, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 8, 0, 0, 0, 0, 0, 0],
        [2, 0, 6, 0, 1, 0, 0, 0, 4],
        [0, 0, 0, 4, 0, 0, 7, 0, 0],
        [1, 0, 4, 0, 6, 0, 0, 0, 7],
        [0, 0, 0, 0, 0, 9, 0, 5, 0],
        [0, 3, 0, 0, 0, 0, 0, 0, 0],
    ],
}


def _recognize_sample_grid(reader: CnnDigitReader, image_name: str) -> list[list[int]]:
    image_path = DATA_IMAGE_ROOT / image_name
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read sample image: {image_path}")

    grid_image = detect_grid_image(image)
    if grid_image is None:
        raise ValueError(f"Failed to detect Sudoku grid from: {image_path}")

    ocr_grid = prepare_grid_for_ocr(grid_image)
    cells = extract_cells(ocr_grid)
    return reader.recognize_grid(cells)


@pytest.fixture(scope="module")
def sample_grids() -> dict[str, list[list[int]]]:
    if not MODEL_PATH.exists():
        pytest.skip(f"CNN model file is missing: {MODEL_PATH}")

    reader = CnnDigitReader(model_path=MODEL_PATH, strict=False)
    if not reader.is_ready:
        pytest.skip(f"CNN model not ready: {reader.load_error}")

    return {
        image_name: _recognize_sample_grid(reader, image_name)
        for image_name in EXPECTED_SAMPLE_GRIDS
    }


def test_recognize_9_in_sudoku_2(sample_grids):
    grid = sample_grids["sudoku_2.png"]
    assert grid[4][7] == 9  # r5c8
    assert grid[8][6] == 9  # r9c7


def test_recognize_9_in_sudoku_3(sample_grids):
    grid = sample_grids["sudoku_3.png"]
    assert grid[0][6] == 9  # r1c7
    assert grid[7][5] == 9  # r8c6


def test_blank_cells_not_false_positive_in_sudoku_3(sample_grids):
    grid = sample_grids["sudoku_3.png"]
    assert grid[5][7] == 0  # r6c8
    assert grid[6][7] == 0  # r7c8


@pytest.mark.parametrize(
    "image_name,minimum_recall",
    [("sudoku_2.png", 35), ("sudoku_3.png", 21)],
)
def test_given_digit_recall_floor(sample_grids, image_name, minimum_recall):
    predicted = sample_grids[image_name]
    expected = EXPECTED_SAMPLE_GRIDS[image_name]

    matched = 0
    total_givens = 0
    for row in range(9):
        for col in range(9):
            truth = expected[row][col]
            if truth == 0:
                continue
            total_givens += 1
            if predicted[row][col] == truth:
                matched += 1

    assert total_givens > 0
    assert matched >= minimum_recall
