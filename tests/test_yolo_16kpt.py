"""Tests for YOLO 16-keypoint detector output parsing."""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.cv.cell_extractor import extract_cells, clean_cell
from backend.cv.detector import warp_grid
from backend.cv.yolo_detector import YoloGridDetector
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.solver.backtracking import SudokuSolver

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestParseKeypoints16:
    def _make_detector(self) -> YoloGridDetector:
        det = object.__new__(YoloGridDetector)
        det._conf_threshold = 0.5
        det._imgsz = 640
        det._session = None
        det._input_name = ""
        det._output_names = []
        det._load_error = None
        det._num_keypoints = 16
        return det

    def test_parse_nms_format_16kpt(self):
        """NMS format: (1, N, 6 + 16*2) = (1, N, 38)."""
        det = self._make_detector()
        row = np.zeros(38, dtype=np.float32)
        row[4] = 0.9
        row[5] = 0
        for i in range(16):
            row[6 + i * 2] = 100.0 + i * 10
            row[6 + i * 2 + 1] = 200.0 + i * 5
        preds = row.reshape(1, 1, 38)
        result = det._parse_best_detection(preds, scale=1.0, pad_x=0, pad_y=0, w_orig=640, h_orig=640)
        assert result is not None
        corners, conf = result
        assert conf == pytest.approx(0.9)
        assert corners.shape == (16, 2)
        assert corners[0, 0] == pytest.approx(100.0)
        assert corners[15, 0] == pytest.approx(250.0)

    def test_parse_non_nms_format_16kpt(self):
        """Non-NMS: (1, 37, N) transposed."""
        det = self._make_detector()
        row = np.zeros(37, dtype=np.float32)
        row[0:4] = [200, 200, 100, 100]
        row[4] = 0.8
        for i in range(16):
            row[5 + i * 2] = 150.0 + i * 5
            row[5 + i * 2 + 1] = 160.0 + i * 3
        preds = row.reshape(1, 37, 1)
        result = det._parse_best_detection(preds, scale=1.0, pad_x=0, pad_y=0, w_orig=640, h_orig=640)
        assert result is not None
        corners, conf = result
        assert conf == pytest.approx(0.8)
        assert corners.shape == (16, 2)

    def test_below_confidence_returns_none(self):
        det = self._make_detector()
        row = np.zeros(38, dtype=np.float32)
        row[4] = 0.3
        preds = row.reshape(1, 1, 38)
        result = det._parse_best_detection(preds, scale=1.0, pad_x=0, pad_y=0, w_orig=640, h_orig=640)
        assert result is None

    def test_letterbox_coord_unmapping(self):
        det = self._make_detector()
        row = np.zeros(38, dtype=np.float32)
        row[4] = 0.9
        row[5] = 0
        row[6] = 320.0
        row[7] = 320.0
        preds = row.reshape(1, 1, 38)
        result = det._parse_best_detection(preds, scale=0.5, pad_x=160.0, pad_y=160.0, w_orig=640, h_orig=480)
        assert result is not None
        corners, _ = result
        assert corners[0, 0] == pytest.approx(320.0)
        assert corners[0, 1] == pytest.approx(320.0)


class TestEndToEnd16Kpt:
    """End-to-end tests: YOLO 16-kpt → adaptive extraction → CNN OCR → solve."""

    @pytest.fixture(scope="class")
    def detector(self):
        model_path = str(REPO_ROOT / "models" / "releases" / "yolo_grid_detect.onnx")
        det = YoloGridDetector(model_path=model_path, num_keypoints=16, conf_threshold=0.3)
        if not det.is_ready:
            pytest.skip(f"YOLO model not available: {det.load_error}")
        return det

    @pytest.fixture(scope="class")
    def reader(self):
        cnn_path = os.environ.get("CNN_MODEL_PATH")
        if not cnn_path:
            pytest.skip("CNN_MODEL_PATH not set")
        r = CnnDigitReader(model_path=cnn_path)
        if not r.is_ready:
            pytest.skip(f"CNN model not available: {r.load_error}")
        return r

    def _run_pipeline(self, img_path: str, detector, reader) -> list[list[int]]:
        """Run full pipeline and return solved grid."""
        image = cv2.imread(img_path)
        assert image is not None, f"Cannot read {img_path}"

        det = detector.detect(image)
        assert det is not None, f"YOLO detection failed for {img_path}"
        assert det.corners.shape == (4, 2), f"Expected 4 corners, got {det.corners.shape}"

        warped = warp_grid(image, det.corners)
        cells = extract_cells(warped)
        assert len(cells) == 81

        cleaned = [clean_cell(c) for c in cells]
        grid, metadata = reader.recognize_grid_with_metadata(cleaned)
        givens = sum(1 for row in grid for c in row if c != 0)
        assert givens >= 17, f"Too few givens: {givens}"

        solver = SudokuSolver()
        solved = solver.solve(grid)
        assert solved is not None, f"Solver failed for {img_path}"
        return solved

    def test_photo_1(self, detector, reader):
        """Validation baseline: photo_1.jpg must be correctly recognized."""
        img = str(REPO_ROOT / "data" / "raw" / "images" / "photo_1.jpg")
        if not Path(img).exists():
            pytest.skip("photo_1.jpg not found")
        solved = self._run_pipeline(img, detector, reader)
        assert all(all(c != 0 for c in row) for row in solved)

    def test_bamboo_grid(self, detector, reader):
        """Decorative grid: IMG_8710.jpg (bamboo-style board)."""
        img = str(REPO_ROOT / "data" / "raw" / "images" / "IMG_8710.jpg")
        if not Path(img).exists():
            pytest.skip("IMG_8710.jpg not found")
        solved = self._run_pipeline(img, detector, reader)
        assert all(all(c != 0 for c in row) for row in solved)
