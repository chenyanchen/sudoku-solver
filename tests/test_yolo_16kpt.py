"""Tests for YOLO 16-keypoint detector output parsing."""
import numpy as np
import pytest
from backend.cv.yolo_detector import YoloGridDetector


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
