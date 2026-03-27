# Phase 1: YOLO 16-Keypoint + Adaptive Cell Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the YOLO grid detector from 4 corner keypoints to 16 keypoints (4×4 block intersections) and build adaptive cell extraction that uses these 16 points for accurate cell splitting on decorative-style sudoku boards.

**Architecture:** The current `YoloGridDetector` returns 4 corners and warps to 450×450. The new detector returns 16 keypoints. A new `extract_cells_adaptive()` function uses those 16 points to perform per-block perspective warp (9 blocks × 150×150), then splits each block into 3×3 cells. The old `extract_cells(warped)` remains untouched for backward compatibility.

**Tech Stack:** Python 3.13, OpenCV, NumPy, ONNX Runtime, ultralytics (training only), Label Studio + ml_backend (labeling)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `backend/cv/yolo_detector.py` | Upgrade to parse 16 keypoints from YOLO output |
| Create | `backend/cv/adaptive_cell_extractor.py` | Two-level warp: 16 kpts → 9 blocks → 81 cells |
| Modify | `data/yolo_grid/sudoku_grid.yaml` | Update `kpt_shape: [16, 2]` |
| Modify | `scripts/auto_label_yolo.py` | Generate 16-kpt labels from 4-corner detections |
| Modify | `scripts/augment_yolo_data.py` | Handle 16-kpt label format in augmentation |
| Modify | `scripts/train_yolo_grid.py` | No code changes needed (reads kpt_shape from YAML) |
| Create | `tests/test_adaptive_cell_extractor.py` | Tests for adaptive cell extraction |
| Create | `tests/test_yolo_16kpt.py` | Tests for 16-kpt detector parsing |

---

### Task 1: Adaptive Cell Extractor — Core Logic

**Files:**
- Create: `backend/cv/adaptive_cell_extractor.py`
- Create: `tests/test_adaptive_cell_extractor.py`

- [ ] **Step 1: Write failing test for `extract_cells_adaptive` with a synthetic grid**

```python
# tests/test_adaptive_cell_extractor.py
"""Tests for adaptive cell extraction using 16 keypoints."""

import numpy as np
import cv2
import pytest

from backend.cv.adaptive_cell_extractor import extract_cells_adaptive


def _make_synthetic_grid(size: int = 900) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic 9x9 grid image with known 16 keypoints.

    Returns (image, keypoints_16) where keypoints are in image coords.
    The grid is drawn starting at (100, 100) with 70px per cell,
    so total grid is 630x630 within a 900x900 image.
    """
    img = np.ones((size, size, 3), dtype=np.uint8) * 200  # gray background

    origin_x, origin_y = 100, 100
    cell_size = 70  # pixels per cell
    block_size = cell_size * 3  # 210px per block

    # Draw grid lines
    for i in range(10):
        y = origin_y + i * cell_size
        x = origin_x + i * cell_size
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(img, (origin_x, y), (origin_x + 9 * cell_size, y), (0, 0, 0), thickness)
        cv2.line(img, (x, origin_y), (x, origin_y + 9 * cell_size), (0, 0, 0), thickness)

    # Fill some cells with digits (draw "5" as a filled region for detection)
    for r in range(9):
        for c in range(9):
            if (r + c) % 5 == 0:  # Sparse pattern
                cx = origin_x + c * cell_size + cell_size // 2
                cy = origin_y + r * cell_size + cell_size // 2
                cv2.putText(img, "5", (cx - 10, cy + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # 16 keypoints: 4x4 grid of block boundary intersections
    keypoints = np.zeros((16, 2), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            idx = row * 4 + col
            keypoints[idx, 0] = origin_x + col * block_size  # x
            keypoints[idx, 1] = origin_y + row * block_size  # y

    return img, keypoints


class TestExtractCellsAdaptive:
    def test_returns_81_cells(self):
        img, kpts = _make_synthetic_grid()
        cells = extract_cells_adaptive(img, kpts)
        assert len(cells) == 81

    def test_cell_shape(self):
        img, kpts = _make_synthetic_grid()
        cells = extract_cells_adaptive(img, kpts)
        for cell in cells:
            assert cell.shape[0] == cell.shape[1], f"Cell not square: {cell.shape}"
            assert cell.shape[0] >= 40, f"Cell too small: {cell.shape}"

    def test_row_major_order(self):
        """Cell 0 is top-left, cell 80 is bottom-right."""
        img, kpts = _make_synthetic_grid()
        cells = extract_cells_adaptive(img, kpts)
        # Cell at index 0 should come from block (0,0), cell (0,0)
        # Cell at index 80 should come from block (2,2), cell (2,2)
        assert len(cells) == 81
        # Verify non-empty (all cells should have content from the synthetic grid)
        assert all(c.size > 0 for c in cells)

    def test_uneven_block_sizes(self):
        """Simulate decorative grid where blocks have different sizes."""
        img = np.ones((900, 900, 3), dtype=np.uint8) * 200

        # Blocks have different widths: 180, 220, 200 (total 600)
        # and different heights: 190, 210, 200 (total 600)
        block_widths = [180, 220, 200]
        block_heights = [190, 210, 200]
        ox, oy = 100, 100

        keypoints = np.zeros((16, 2), dtype=np.float32)
        cum_x = [ox]
        for w in block_widths:
            cum_x.append(cum_x[-1] + w)
        cum_y = [oy]
        for h in block_heights:
            cum_y.append(cum_y[-1] + h)

        for row in range(4):
            for col in range(4):
                keypoints[row * 4 + col] = [cum_x[col], cum_y[row]]

        cells = extract_cells_adaptive(img, keypoints)
        assert len(cells) == 81
        # All cells should still be square and usable
        for cell in cells:
            assert cell.shape[0] == cell.shape[1]
            assert cell.shape[0] >= 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_adaptive_cell_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.cv.adaptive_cell_extractor'`

- [ ] **Step 3: Implement `extract_cells_adaptive`**

```python
# backend/cv/adaptive_cell_extractor.py
"""Adaptive cell extraction using 16 keypoints (4x4 block intersections).

Two-level warp: 16 keypoints define 9 block quadrilaterals, each warped
independently to 150x150, then split into 3x3 cells of 50x50.
This handles decorative grids where block separators have uneven widths.
"""

from __future__ import annotations

import cv2
import numpy as np

# Each cell is extracted at this size from the 150x150 block warp.
_BLOCK_WARP_SIZE = 150
_CELL_SIZE = 50  # 150 / 3
_BORDER_SKIP = 1  # pixels to skip for grid line removal


def extract_cells_adaptive(
    image: np.ndarray,
    keypoints_16: np.ndarray,
    cell_output_size: int = 48,
) -> list[np.ndarray]:
    """Extract 81 cells from an image using 16 keypoints.

    Args:
        image: Original image (any size, BGR or grayscale).
        keypoints_16: Shape (16, 2), YOLO-detected keypoints in image coords.
                      Arranged as 4x4 grid (row-major): indices 0-3 = top row,
                      4-7 = second row, 8-11 = third row, 12-15 = bottom row.
        cell_output_size: Output cell size (default 48 to match CNN input).

    Returns:
        81 cell images in row-major order (top-left to bottom-right).
    """
    kpts = keypoints_16.reshape(4, 4, 2).astype(np.float32)

    dst_block = np.float32([
        [0, 0],
        [_BLOCK_WARP_SIZE - 1, 0],
        [_BLOCK_WARP_SIZE - 1, _BLOCK_WARP_SIZE - 1],
        [0, _BLOCK_WARP_SIZE - 1],
    ])

    cells: list[np.ndarray] = []

    for block_row in range(3):
        for block_col in range(3):
            # Four corners of this block from the keypoint grid
            src_quad = np.float32([
                kpts[block_row, block_col],          # TL
                kpts[block_row, block_col + 1],      # TR
                kpts[block_row + 1, block_col + 1],  # BR
                kpts[block_row + 1, block_col],      # BL
            ])

            matrix = cv2.getPerspectiveTransform(src_quad, dst_block)
            block_img = cv2.warpPerspective(image, matrix, (_BLOCK_WARP_SIZE, _BLOCK_WARP_SIZE))

            # Split block into 3x3 cells
            for cell_row in range(3):
                for cell_col in range(3):
                    x = cell_col * _CELL_SIZE
                    y = cell_row * _CELL_SIZE
                    # Skip border pixels to remove grid lines
                    cell = block_img[
                        y + _BORDER_SKIP : y + _CELL_SIZE - _BORDER_SKIP,
                        x + _BORDER_SKIP : x + _CELL_SIZE - _BORDER_SKIP,
                    ]
                    # Resize to target output size
                    if cell_output_size != cell.shape[0]:
                        cell = cv2.resize(cell, (cell_output_size, cell_output_size))
                    cells.append(cell)

    return cells
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_adaptive_cell_extractor.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/cv/adaptive_cell_extractor.py tests/test_adaptive_cell_extractor.py
git commit -m "feat: add adaptive cell extraction using 16 keypoints"
```

---

### Task 2: Upgrade YoloGridDetector to 16 Keypoints

**Files:**
- Modify: `backend/cv/yolo_detector.py`
- Create: `tests/test_yolo_16kpt.py`

- [ ] **Step 1: Write failing test for 16-keypoint parsing**

```python
# tests/test_yolo_16kpt.py
"""Tests for YOLO 16-keypoint detector output parsing."""

import numpy as np
import pytest

from backend.cv.yolo_detector import YoloGridDetector


class TestParseKeypoints16:
    """Test _parse_best_detection with 16-keypoint output format."""

    def _make_detector(self) -> YoloGridDetector:
        """Create detector without loading a model (for unit-testing parse logic)."""
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
        # Build a single detection: [x1, y1, x2, y2, conf, class_id, kx0, ky0, ..., kx15, ky15]
        row = np.zeros(38, dtype=np.float32)
        row[0:4] = [100, 100, 300, 300]  # bbox
        row[4] = 0.9  # confidence
        row[5] = 0  # class id
        # 16 keypoints at known positions (in letterbox coords)
        for i in range(16):
            row[6 + i * 2] = 100.0 + i * 10  # kx
            row[6 + i * 2 + 1] = 200.0 + i * 5  # ky

        preds = row.reshape(1, 1, 38)
        result = det._parse_best_detection(preds, scale=1.0, pad_x=0, pad_y=0, w_orig=640, h_orig=640)

        assert result is not None
        corners, conf = result
        assert conf == pytest.approx(0.9)
        assert corners.shape == (16, 2)
        assert corners[0, 0] == pytest.approx(100.0)
        assert corners[0, 1] == pytest.approx(200.0)
        assert corners[15, 0] == pytest.approx(250.0)
        assert corners[15, 1] == pytest.approx(275.0)

    def test_parse_non_nms_format_16kpt(self):
        """Non-NMS format: (1, 5 + 16*2, N) transposed to (N, 37)."""
        det = self._make_detector()
        # [cx, cy, w, h, cls_score, kx0, ky0, ..., kx15, ky15]
        row = np.zeros(37, dtype=np.float32)
        row[0:4] = [200, 200, 100, 100]  # cx, cy, w, h
        row[4] = 0.8  # cls_score
        for i in range(16):
            row[5 + i * 2] = 150.0 + i * 5
            row[5 + i * 2 + 1] = 160.0 + i * 3

        # Simulate (1, 37, 1) which gets transposed
        preds = row.reshape(1, 37, 1)
        result = det._parse_best_detection(preds, scale=1.0, pad_x=0, pad_y=0, w_orig=640, h_orig=640)

        assert result is not None
        corners, conf = result
        assert conf == pytest.approx(0.8)
        assert corners.shape == (16, 2)

    def test_below_confidence_returns_none(self):
        det = self._make_detector()
        row = np.zeros(38, dtype=np.float32)
        row[4] = 0.3  # below threshold
        preds = row.reshape(1, 1, 38)
        result = det._parse_best_detection(preds, scale=1.0, pad_x=0, pad_y=0, w_orig=640, h_orig=640)
        assert result is None

    def test_letterbox_coord_unmapping(self):
        """Verify keypoints are correctly mapped back from letterbox coords."""
        det = self._make_detector()
        row = np.zeros(38, dtype=np.float32)
        row[4] = 0.9
        row[5] = 0
        # Place first keypoint at letterbox (320, 320) with scale=0.5, pad=(160, 160)
        row[6] = 320.0  # kx
        row[7] = 320.0  # ky
        preds = row.reshape(1, 1, 38)

        result = det._parse_best_detection(
            preds, scale=0.5, pad_x=160.0, pad_y=160.0, w_orig=640, h_orig=480,
        )
        assert result is not None
        corners, _ = result
        # orig_x = (320 - 160) / 0.5 = 320
        # orig_y = (320 - 160) / 0.5 = 320
        assert corners[0, 0] == pytest.approx(320.0)
        assert corners[0, 1] == pytest.approx(320.0)

    def test_detect_returns_keypoints_and_warped(self):
        """Full detect() should return (warped, keypoints_16, conf) when model is loaded."""
        # This test requires an actual ONNX model; skip if not available.
        import os
        model_path = os.environ.get("YOLO_16KPT_MODEL")
        if not model_path:
            pytest.skip("YOLO_16KPT_MODEL not set")
        det = YoloGridDetector(model_path=model_path, num_keypoints=16)
        if not det.is_ready:
            pytest.skip(f"Model not loaded: {det.load_error}")
        img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        # Just verify it doesn't crash; detection on blank image may return None
        result = det.detect(img)
        assert result is None or len(result) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_yolo_16kpt.py -v`
Expected: FAIL — `_num_keypoints` attribute doesn't exist, `_parse_best_detection` returns 4-point shape

- [ ] **Step 3: Modify `YoloGridDetector` to support configurable keypoint count**

Replace the entire `backend/cv/yolo_detector.py` with the upgraded version. Key changes:
- Add `num_keypoints` parameter (default 4 for backward compat)
- `_parse_best_detection` returns `(N, 2)` keypoints where N = `num_keypoints`
- `detect()` returns `(warped, keypoints, conf)` — when `num_keypoints=4`, warp uses 4 corners as before; when `num_keypoints=16`, warp uses the 4 outer corners (indices 0, 3, 12, 15)

The modified `_parse_best_detection` in `backend/cv/yolo_detector.py`:

```python
"""YOLO26n-pose grid detector: detects sudoku grid + corner/block keypoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

DEFAULT_YOLO_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "releases"
    / "yolo_grid_detect.onnx"
)

# Indices of the 4 outer corners within the 16-keypoint layout:
#  0 ── 1 ── 2 ── 3
#  │    │    │    │
#  4 ── 5 ── 6 ── 7
#  │    │    │    │
#  8 ── 9 ──10 ──11
#  │    │    │    │
# 12 ──13 ──14 ──15
_OUTER_CORNER_INDICES_16 = [0, 3, 15, 12]  # TL, TR, BR, BL


class YoloGridDetector:
    """Detect sudoku grids using YOLO26n-pose ONNX model.

    Supports both 4-keypoint (corners only) and 16-keypoint (4x4 block
    intersections) models. Returns keypoints in the original image
    coordinate space.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        imgsz: int = 640,
        num_keypoints: int = 4,
    ) -> None:
        self._conf_threshold = conf_threshold
        self._imgsz = imgsz
        self._num_keypoints = num_keypoints
        self._session = None
        self._input_name = ""
        self._output_names: list[str] = []
        self._load_error: Optional[str] = None

        path = Path(model_path) if model_path else DEFAULT_YOLO_MODEL
        self._load_model(path)

    @property
    def is_ready(self) -> bool:
        return self._session is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def num_keypoints(self) -> int:
        return self._num_keypoints

    def detect(
        self, image: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
        """Detect the best sudoku grid in an image.

        Args:
            image: BGR image (any size).

        Returns:
            (warped_450x450, keypoints_Nx2, confidence) or None.
            N = num_keypoints (4 or 16).
        """
        if not self.is_ready:
            return None

        h_orig, w_orig = image.shape[:2]
        img_input, scale, pad_x, pad_y = self._preprocess(image)

        outputs = self._session.run(
            self._output_names,
            {self._input_name: img_input},
        )
        preds = outputs[0]

        best = self._parse_best_detection(preds, scale, pad_x, pad_y, w_orig, h_orig)
        if best is None:
            return None

        keypoints, conf = best

        # Warp using outer 4 corners.
        if self._num_keypoints == 4:
            corners_4 = keypoints  # already (4, 2)
        else:
            corners_4 = keypoints[_OUTER_CORNER_INDICES_16]  # (4, 2)

        dst = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
        matrix = cv2.getPerspectiveTransform(corners_4.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, matrix, (450, 450))

        return warped, keypoints, conf

    def _load_model(self, path: Path) -> None:
        if ort is None:
            self._load_error = "onnxruntime not installed"
            return
        if not path.exists():
            self._load_error = f"YOLO model not found: {path}"
            return
        try:
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(path), opts, providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            LOGGER.info("YOLO grid detector loaded: %s (%d keypoints)", path.name, self._num_keypoints)
        except Exception as e:
            self._load_error = f"Failed to load YOLO model: {e}"

    def _preprocess(
        self, image: np.ndarray,
    ) -> tuple[np.ndarray, float, float, float]:
        """Letterbox resize + normalize for YOLO input."""
        h, w = image.shape[:2]
        scale = min(self._imgsz / w, self._imgsz / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.full(
            (self._imgsz, self._imgsz, 3), 114, dtype=np.uint8,
        )
        pad_x = (self._imgsz - new_w) / 2
        pad_y = (self._imgsz - new_h) / 2
        x0 = int(pad_x)
        y0 = int(pad_y)
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[None], scale, pad_x, pad_y

    def _parse_best_detection(
        self,
        preds: np.ndarray,
        scale: float,
        pad_x: float,
        pad_y: float,
        w_orig: int,
        h_orig: int,
    ) -> Optional[tuple[np.ndarray, float]]:
        """Parse YOLO output and return best (keypoints_Nx2, confidence)."""
        raw = np.squeeze(preds, axis=0)

        if raw.ndim == 2 and raw.shape[0] < raw.shape[1]:
            raw = raw.T

        n_kpt = self._num_keypoints
        # NMS format: 6 + n_kpt * 2 columns
        # Non-NMS format: 5 + n_kpt * 2 columns
        min_cols_nms = 6 + n_kpt * 2
        min_cols_non_nms = 5 + n_kpt * 2

        if raw.ndim != 2 or raw.shape[1] < min_cols_non_nms:
            return None

        ncols = raw.shape[1]

        if ncols >= min_cols_nms:
            conf = raw[:, 4]
            kpt_start = 6
        elif ncols >= min_cols_non_nms:
            conf = raw[:, 4]
            kpt_start = 5
        else:
            return None

        mask = conf >= self._conf_threshold
        if not np.any(mask):
            return None

        indices = np.where(mask)[0]
        best_idx = indices[np.argmax(conf[mask])]
        best_conf = float(conf[best_idx])

        kpts = []
        for i in range(n_kpt):
            kx = float(raw[best_idx, kpt_start + i * 2])
            ky = float(raw[best_idx, kpt_start + i * 2 + 1])
            orig_x = (kx - pad_x) / scale
            orig_y = (ky - pad_y) / scale
            orig_x = max(0.0, min(orig_x, w_orig - 1))
            orig_y = max(0.0, min(orig_y, h_orig - 1))
            kpts.append([orig_x, orig_y])

        return np.float32(kpts), best_conf
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_yolo_16kpt.py -v -k "not detect_returns"`
Expected: 4 tests PASS (the `detect_returns` test is skipped without model)

- [ ] **Step 5: Run existing tests to verify backward compatibility**

Run: `uv run pytest tests/test_cv.py -v`
Expected: All existing tests PASS (default `num_keypoints=4` preserves old behavior)

- [ ] **Step 6: Commit**

```bash
git add backend/cv/yolo_detector.py tests/test_yolo_16kpt.py
git commit -m "feat: upgrade YoloGridDetector to support 16 keypoints"
```

---

### Task 3: Update Label Generation for 16 Keypoints

**Files:**
- Modify: `scripts/auto_label_yolo.py`
- Modify: `data/yolo_grid/sudoku_grid.yaml`

- [ ] **Step 1: Update `sudoku_grid.yaml` to 16 keypoints**

```yaml
# data/yolo_grid/sudoku_grid.yaml
# Absolute path set to avoid ultralytics CWD resolution issues
path: /Users/nle/go/src/github.com/chenyanchen/sudoku-solver/data/yolo_grid
train: images/train
val: images/train

names:
  0: sudoku_grid

kpt_shape: [16, 2]
flip_idx: []
```

- [ ] **Step 2: Add 16-keypoint generation to `auto_label_yolo.py`**

Add a new function `corners_to_16kpt_yolo_pose` that takes 4 corners, warps to 450×450, and interpolates the remaining 12 keypoints from the known grid geometry. Modify `main()` to call this instead of `corners_to_yolo_pose`.

Add after the existing `corners_to_yolo_pose` function:

```python
def corners_4_to_keypoints_16(corners: np.ndarray) -> np.ndarray:
    """Interpolate 4 corners (TL, TR, BR, BL) into 16 block-intersection keypoints.

    The 16 points form a 4x4 grid corresponding to the intersections of the
    3x3 block boundaries. Points are evenly spaced along the edges of the
    quadrilateral defined by the 4 corners using bilinear interpolation.

    Args:
        corners: Shape (4, 2), order [TL, TR, BR, BL].

    Returns:
        Shape (16, 2) keypoints in the same coordinate space as input corners.
    """
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

    keypoints = np.zeros((16, 2), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            # Bilinear interpolation parameter
            u = col / 3.0  # 0, 1/3, 2/3, 1
            v = row / 3.0
            # Interpolate along top and bottom edges
            top = tl * (1 - u) + tr * u
            bot = bl * (1 - u) + br * u
            # Interpolate between top and bottom
            pt = top * (1 - v) + bot * v
            keypoints[row * 4 + col] = pt

    return keypoints


def keypoints_16_to_yolo_pose(keypoints: np.ndarray, img_w: int, img_h: int) -> str:
    """Convert 16 keypoints to YOLO pose label line.

    Format: <cls> <cx> <cy> <bw> <bh> <kx0> <ky0> ... <kx15> <ky15>
    All values normalized to [0, 1].
    """
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(16):
        kx = float(keypoints[i, 0]) / img_w
        ky = float(keypoints[i, 1]) / img_h
        parts.append(f"{kx:.6f} {ky:.6f}")

    return " ".join(parts)
```

Then in the `main()` function, replace the label generation line:

```python
# OLD:
label = corners_to_yolo_pose(corners, w, h)

# NEW:
kpts_16 = corners_4_to_keypoints_16(corners)
label = keypoints_16_to_yolo_pose(kpts_16, w, h)
```

- [ ] **Step 3: Verify the script runs without error on existing images**

Run: `uv run python scripts/auto_label_yolo.py --src data/raw/images --dst /tmp/yolo_16kpt_test`
Expected: Labels generated successfully. Each label file should have 1 + 4 + 32 = 37 values (class + bbox + 16×2 keypoints).

```bash
# Verify label format
head -1 /tmp/yolo_16kpt_test/labels/train/*.txt | head -5
# Each line should have 37 space-separated values
awk '{print NF}' /tmp/yolo_16kpt_test/labels/train/*.txt | sort -u
# Expected output: 37
```

- [ ] **Step 4: Commit**

```bash
git add scripts/auto_label_yolo.py data/yolo_grid/sudoku_grid.yaml
git commit -m "feat: generate 16-keypoint labels from 4-corner detections"
```

---

### Task 4: Update Data Augmentation for 16 Keypoints

**Files:**
- Modify: `scripts/augment_yolo_data.py`

- [ ] **Step 1: Update `load_source_grids` and augmentation to handle 16 keypoints**

The current `load_source_grids` reads 4 keypoints from label files (indices 5-12). Update it to handle 16 keypoints (indices 5-36). The `corners_to_yolo_pose` function in this file also needs updating to output 16 keypoints.

In `augment_yolo_data.py`, make these changes:

1. Update `corners_to_yolo_pose` to accept and output 16 keypoints:

```python
def corners_to_yolo_pose(keypoints: np.ndarray, img_w: int, img_h: int) -> str:
    """Convert keypoints (4 or 16) to YOLO pose format label."""
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(len(keypoints)):
        kx = float(keypoints[i, 0]) / img_w
        ky = float(keypoints[i, 1]) / img_h
        parts.append(f"{kx:.6f} {ky:.6f}")

    return " ".join(parts)
```

2. Update `load_source_grids` to read 16 keypoints and return both grids and keypoints:

```python
def load_source_grids(label_dir: Path, image_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load grid regions and their 16 keypoints from labeled images.

    Returns list of (grid_450x450, keypoints_16_normalized) tuples.
    """
    results = []
    for lbl_path in sorted(label_dir.glob("*.txt")):
        # Try .png then other extensions
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = image_dir / f"{lbl_path.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        line = lbl_path.read_text().strip()
        parts = list(map(float, line.split()))
        h, w = image.shape[:2]

        n_values = len(parts)
        if n_values >= 37:  # 16 keypoints: 5 + 16*2
            n_kpt = 16
        elif n_values >= 13:  # 4 keypoints: 5 + 4*2
            n_kpt = 4
        else:
            continue

        keypoints = np.zeros((n_kpt, 2), dtype=np.float32)
        for i in range(n_kpt):
            keypoints[i, 0] = parts[5 + i * 2] * w
            keypoints[i, 1] = parts[5 + i * 2 + 1] * h

        # Extract grid using outer 4 corners
        if n_kpt == 16:
            corners_4 = keypoints[[0, 3, 15, 12]]  # TL, TR, BR, BL
        else:
            corners_4 = keypoints  # already [TL, TR, BR, BL]

        grid = extract_grid_region(image, corners_4)
        # Store normalized keypoints (relative to grid corners) for re-projection
        results.append((grid, keypoints / np.array([w, h], dtype=np.float32)))

    return results
```

3. Update the main loop to generate 16-keypoint labels for augmented images. After pasting the grid with `random_quad` corners, interpolate 16 keypoints from the 4 corners:

```python
# In main(), replace the label generation:
# OLD:
# label = corners_to_yolo_pose(corners, w, h)

# NEW:
from scripts.auto_label_yolo import corners_4_to_keypoints_16
kpts_16 = corners_4_to_keypoints_16(corners)
label = corners_to_yolo_pose(kpts_16, w, h)
```

Since importing from `scripts.auto_label_yolo` may not work cleanly, copy the `corners_4_to_keypoints_16` function into this file instead (it's 15 lines of bilinear interpolation):

```python
def corners_4_to_keypoints_16(corners: np.ndarray) -> np.ndarray:
    """Interpolate 4 corners [TL, TR, BR, BL] into 16 block-intersection keypoints."""
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
    keypoints = np.zeros((16, 2), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            u = col / 3.0
            v = row / 3.0
            top = tl * (1 - u) + tr * u
            bot = bl * (1 - u) + br * u
            keypoints[row * 4 + col] = top * (1 - v) + bot * v
    return keypoints
```

Then update the `main()` function — `load_source_grids` now returns tuples, and label generation uses 16 keypoints:

```python
# OLD:
grids = load_source_grids(lbl_dir, img_dir)
# ...
grid = augment_grid(random.choice(grids))
# ...
label = corners_to_yolo_pose(corners, w, h)

# NEW:
source_data = load_source_grids(lbl_dir, img_dir)
if not source_data:
    print("No source grids found. Run auto_label_yolo.py first.")
    return 1
print(f"Loaded {len(source_data)} source grids")
# ...
grid, _ = random.choice(source_data)
grid = augment_grid(grid)
# ...
kpts_16 = corners_4_to_keypoints_16(corners)
label = corners_to_yolo_pose(kpts_16, w, h)
```

- [ ] **Step 2: Verify augmentation generates correct 16-kpt labels**

Run: `uv run python scripts/augment_yolo_data.py --count 5 --dst /tmp/yolo_16kpt_aug_test`
Expected: 5 augmented images with 37-value labels.

```bash
awk '{print NF}' /tmp/yolo_16kpt_aug_test/labels/train/aug_*.txt | sort -u
# Expected: 37
```

- [ ] **Step 3: Commit**

```bash
git add scripts/augment_yolo_data.py
git commit -m "feat: update data augmentation for 16-keypoint labels"
```

---

### Task 5: Regenerate Dataset and Train 16-Keypoint Model

This task involves running the full training pipeline. It produces a new ONNX model.

**Files:**
- Output: `data/yolo_grid/labels/train/*.txt` (regenerated)
- Output: `data/yolo_grid/images/train/*.png` (augmented)
- Output: `models/releases/yolo_grid_detect.onnx` (new 16-kpt model)

- [ ] **Step 1: Regenerate base labels from raw images**

```bash
# Clear old labels and images (augmented ones will be regenerated)
rm -f data/yolo_grid/labels/train/aug_*.txt
rm -f data/yolo_grid/images/train/aug_*.png

# Regenerate base labels with 16 keypoints
uv run python scripts/auto_label_yolo.py --src data/raw/images --dst data/yolo_grid
```

Expected: Base labels regenerated with 37 values each.

- [ ] **Step 2: Regenerate augmented dataset**

```bash
uv run python scripts/augment_yolo_data.py --count 500 --dst data/yolo_grid
```

Expected: ~500 augmented images + labels added to the dataset.

- [ ] **Step 3: Train the 16-keypoint model**

```bash
uv run python scripts/train_yolo_grid.py --epochs 100 --model yolo26n-pose.pt
```

Expected: Training completes. Best model exported to `models/yolo_runs/grid_detect/weights/best.pt` and ONNX to `models/releases/yolo_grid_detect.onnx`.

Monitor: pose loss should decrease steadily. Final mAP@50 (pose) should be > 0.95.

- [ ] **Step 4: Verify the new model has 16-keypoint output**

```bash
uv run python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('models/releases/yolo_grid_detect.onnx')
inp = np.random.rand(1, 3, 640, 640).astype(np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: inp})
print(f'Output shape: {out[0].shape}')
# For 16 keypoints, expect last dim to be 37 (non-NMS) or 38 (NMS)
"
```

- [ ] **Step 5: Commit the new model and dataset config**

```bash
git add data/yolo_grid/sudoku_grid.yaml models/releases/yolo_grid_detect.onnx
git commit -m "feat: train YOLO 16-keypoint grid detection model"
```

Note: Do not commit the augmented training images (`data/yolo_grid/images/train/aug_*.png`) or training artifacts (`models/yolo_runs/`) to git — they are large and reproducible.

---

### Task 6: End-to-End Integration Test

**Files:**
- Modify: `tests/test_yolo_16kpt.py` (add integration test)

- [ ] **Step 1: Write end-to-end test using real images**

Add to `tests/test_yolo_16kpt.py`:

```python
import os
from pathlib import Path

import cv2
import numpy as np

from backend.cv.yolo_detector import YoloGridDetector
from backend.cv.adaptive_cell_extractor import extract_cells_adaptive
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.solver.backtracking import SudokuSolver


REPO_ROOT = Path(__file__).resolve().parents[1]


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

        result = detector.detect(image)
        assert result is not None, f"YOLO detection failed for {img_path}"
        warped, keypoints, conf = result

        assert keypoints.shape == (16, 2), f"Expected 16 keypoints, got {keypoints.shape}"

        cells = extract_cells_adaptive(image, keypoints)
        assert len(cells) == 81

        from backend.cv.cell_extractor import clean_cell
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
```

- [ ] **Step 2: Run the end-to-end test**

Run: `CNN_MODEL_PATH=models/releases/sudoku_digit_cnn_v3.4.onnx uv run pytest tests/test_yolo_16kpt.py::TestEndToEnd16Kpt -v`
Expected: Both tests PASS (photo_1.jpg and IMG_8710.jpg correctly detected, OCR'd, and solved).

If either test fails, investigate:
- YOLO detection failure → check model training quality, may need more epochs or data
- OCR failure → check cell extraction quality by saving intermediate cell images
- Solver failure → check OCR grid output for misrecognized digits

- [ ] **Step 3: Commit**

```bash
git add tests/test_yolo_16kpt.py
git commit -m "test: add end-to-end integration tests for 16-keypoint pipeline"
```

---

### Task 7: Update ML Backend for 16 Keypoints

**Files:**
- Modify: `ml_backend/app.py`

- [ ] **Step 1: Update `_get_yolo()` to use 16 keypoints**

In `ml_backend/app.py`, modify the `_get_yolo` function:

```python
def _get_yolo():
    global _yolo
    if _yolo is None:
        from backend.cv.yolo_detector import YoloGridDetector
        _yolo = YoloGridDetector(
            model_path=os.getenv("YOLO_MODEL"),
            num_keypoints=16,
        )
        if _yolo.is_ready:
            logger.info("YOLO loaded (16-kpt)")
        else:
            logger.error("YOLO failed: %s", _yolo.load_error)
    return _yolo
```

- [ ] **Step 2: Update the `/predict` endpoint to return 16 keypoints**

Find the section in `app.py` where YOLO detection results are converted to Label Studio keypoint annotations. Update it to output 16 keypoints instead of 4. The exact changes depend on how the predict endpoint formats keypoints — read the current code and update the keypoint iteration from `range(4)` to `range(16)`, and update any hardcoded corner labels (TL/TR/BR/BL) to the 16-point naming scheme.

- [ ] **Step 3: Update the `/fit` endpoint to save 16-keypoint labels**

The `/fit` endpoint saves submitted annotations back to `data/yolo_grid/`. Update it to read 16 keypoints from Label Studio's annotation format and write 37-value YOLO label lines.

- [ ] **Step 4: Smoke test the ML backend**

```bash
CNN_MODEL_PATH=models/releases/sudoku_digit_cnn_v3.4.onnx uv run python -m ml_backend --port 9090 &
sleep 2
curl -s http://localhost:9090/health | python -m json.tool
# Expected: {"status": "ok"}
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add ml_backend/app.py
git commit -m "feat: update ML backend to use 16-keypoint YOLO model"
```
