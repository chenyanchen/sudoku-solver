# Detector Abstraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Abstract grid detection behind a `GridDetector` Protocol with `DetectionResult(corners, detector_name)`, decouple warp from detection, and wire up web app + camera monitor.

**Architecture:** A `GridDetector` Protocol defines the contract (`detect → DetectionResult | None`). Three implementations: `YoloGridDetector`, `OpenCVGridDetector`, `ChainDetector`. `warp_grid()` is a standalone function. Web app uses `ChainDetector(yolo, opencv)`. Camera monitor uses `YoloGridDetector` directly. Screen monitor is deferred (its multi-candidate pipeline is architecturally different from single-result detection).

**Tech Stack:** Python 3.13, OpenCV, NumPy, Protocol (typing)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `backend/cv/detector.py` | `DetectionResult`, `GridDetector` Protocol, `warp_grid()`, `ChainDetector` |
| Modify | `backend/cv/yolo_detector.py` | `detect()` returns `DetectionResult`, remove warp logic |
| Create | `backend/cv/opencv_detector.py` | `OpenCVGridDetector` wrapping existing `grid_detector.py` with CLAHE retry |
| Modify | `backend/models/schemas.py` | Add `detector_name` field to `ImageSolveResponse` |
| Modify | `backend/api/routes.py` | Use `ChainDetector`, pass `detector_name` to response |
| Modify | `camera_monitor/app.py` | Handle new `DetectionResult` return type |
| Create | `tests/test_detector.py` | Tests for Protocol, `warp_grid()`, `ChainDetector` |

---

### Task 1: Core Abstractions — `detector.py`

**Files:**
- Create: `backend/cv/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_detector.py
"""Tests for grid detector abstractions."""

import numpy as np
import cv2

from backend.cv.detector import DetectionResult, ChainDetector, warp_grid


class TestDetectionResult:
    def test_fields(self):
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        result = DetectionResult(corners=corners, detector_name="test")
        assert result.detector_name == "test"
        assert result.corners.shape == (4, 2)


class TestWarpGrid:
    def test_warps_to_450(self):
        img = np.ones((500, 500, 3), dtype=np.uint8) * 128
        corners = np.array([[50, 50], [450, 50], [450, 450], [50, 450]], dtype=np.float32)
        warped = warp_grid(img, corners)
        assert warped.shape == (450, 450, 3)

    def test_custom_size(self):
        img = np.ones((500, 500, 3), dtype=np.uint8) * 128
        corners = np.array([[50, 50], [450, 50], [450, 450], [50, 450]], dtype=np.float32)
        warped = warp_grid(img, corners, size=300)
        assert warped.shape == (300, 300, 3)


class TestChainDetector:
    def test_returns_first_success(self):
        """ChainDetector returns the first non-None result."""

        class AlwaysNone:
            def detect(self, image):
                return None

        class AlwaysFound:
            def detect(self, image):
                corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
                return DetectionResult(corners=corners, detector_name="found")

        chain = ChainDetector([AlwaysNone(), AlwaysFound()])
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = chain.detect(img)
        assert result is not None
        assert result.detector_name == "found"

    def test_returns_none_when_all_fail(self):
        class AlwaysNone:
            def detect(self, image):
                return None

        chain = ChainDetector([AlwaysNone(), AlwaysNone()])
        result = chain.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert result is None

    def test_stops_at_first_success(self):
        """Should not call later detectors after a success."""
        call_count = [0, 0]

        class Detector0:
            def detect(self, image):
                call_count[0] += 1
                corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
                return DetectionResult(corners=corners, detector_name="d0")

        class Detector1:
            def detect(self, image):
                call_count[1] += 1
                return None

        chain = ChainDetector([Detector0(), Detector1()])
        chain.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        assert call_count == [1, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.cv.detector'`

- [ ] **Step 3: Implement `detector.py`**

```python
# backend/cv/detector.py
"""Grid detector abstractions: Protocol, result type, utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """Result of grid detection: 4 corner coordinates + which detector found it."""

    corners: np.ndarray  # (4, 2), [TL, TR, BR, BL] in original image coords
    detector_name: str


@runtime_checkable
class GridDetector(Protocol):
    """Protocol for grid detectors."""

    def detect(self, image: np.ndarray) -> DetectionResult | None: ...


def warp_grid(image: np.ndarray, corners: np.ndarray, size: int = 450) -> np.ndarray:
    """Perspective-warp a grid region to a square image.

    Args:
        image: Source image (BGR).
        corners: (4, 2) array of [TL, TR, BR, BL] in image coords.
        size: Output square side length (default 450).

    Returns:
        Warped square image.
    """
    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (size, size))


class ChainDetector:
    """Try multiple detectors in order, return the first successful result."""

    def __init__(self, detectors: list[GridDetector]) -> None:
        self._detectors = detectors

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        for d in self._detectors:
            result = d.detect(image)
            if result is not None:
                return result
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_detector.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/cv/detector.py tests/test_detector.py
git commit -m "feat: add GridDetector Protocol, DetectionResult, warp_grid, ChainDetector"
```

---

### Task 2: Update `YoloGridDetector` to Return `DetectionResult`

**Files:**
- Modify: `backend/cv/yolo_detector.py`

- [ ] **Step 1: Update `detect()` method**

Change the `detect()` method to return `DetectionResult | None` instead of `tuple[np.ndarray, np.ndarray, float] | None`. Remove warp logic from `detect()`. Store confidence as `last_confidence` for consumers that need it.

In `backend/cv/yolo_detector.py`, add import at the top (after `import numpy as np`):

```python
from .detector import DetectionResult
```

Replace the `detect()` method (lines 71-118) with:

```python
    def detect(
        self, image: np.ndarray,
    ) -> DetectionResult | None:
        """Detect the best sudoku grid in an image.

        Args:
            image: BGR image (any size).

        Returns:
            DetectionResult with 4 corner coordinates, or None.
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
            self.last_confidence = 0.0
            return None

        keypoints, conf = best
        self.last_confidence = conf

        # Extract the 4 outer corners.
        if self._num_keypoints == 16:
            corners = keypoints[_OUTER_CORNER_INDICES_16]
        else:
            corners = keypoints

        return DetectionResult(corners=corners, detector_name="yolo")
```

Also add `self.last_confidence: float = 0.0` in `__init__` (after line 54, `self._load_error`).

- [ ] **Step 2: Run existing YOLO tests**

Run: `uv run pytest tests/test_yolo_16kpt.py::TestParseKeypoints16 -v`
Expected: All 4 parsing tests PASS (they test `_parse_best_detection` directly, not `detect()`)

- [ ] **Step 3: Commit**

```bash
git add backend/cv/yolo_detector.py
git commit -m "refactor: YoloGridDetector.detect() returns DetectionResult, warp removed"
```

---

### Task 3: Create `OpenCVGridDetector`

**Files:**
- Create: `backend/cv/opencv_detector.py`

- [ ] **Step 1: Implement `OpenCVGridDetector`**

```python
# backend/cv/opencv_detector.py
"""OpenCV contour-based grid detector, wrapped as GridDetector."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import DetectionResult
from .grid_detector import find_grid_with_corners


class OpenCVGridDetector:
    """Detect sudoku grids using OpenCV contour detection.

    Tries direct detection first, then retries with CLAHE enhancement.
    """

    def detect(self, image: np.ndarray) -> DetectionResult | None:
        # Direct detection.
        result = find_grid_with_corners(image)
        if result is not None:
            _, corners = result
            return DetectionResult(corners=corners, detector_name="opencv")

        # CLAHE-enhanced retry.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        result = find_grid_with_corners(enhanced)
        if result is not None:
            _, corners = result
            return DetectionResult(corners=corners, detector_name="opencv")

        return None
```

- [ ] **Step 2: Commit**

```bash
git add backend/cv/opencv_detector.py
git commit -m "feat: add OpenCVGridDetector wrapping contour detection + CLAHE retry"
```

---

### Task 4: Update Web App Routes + Schema

**Files:**
- Modify: `backend/models/schemas.py`
- Modify: `backend/api/routes.py`

- [ ] **Step 1: Add `detector_name` to `ImageSolveResponse`**

In `backend/models/schemas.py`, add a new field to `ImageSolveResponse` (after the `ocr_latency_ms` field):

```python
    detector_name: str | None = Field(
        default=None, description="Grid detector that found the puzzle (e.g. yolo, opencv)"
    )
```

- [ ] **Step 2: Rewrite detection logic in `routes.py`**

Replace the imports and detection functions. The key changes:
- Remove `find_grid` and `YoloGridDetector` imports
- Import `ChainDetector`, `warp_grid`, `DetectionResult` from `detector.py`
- Import `YoloGridDetector` and `OpenCVGridDetector` for initialization
- Replace `_get_yolo_detector()` and `detect_grid_image()` with a `ChainDetector` singleton

Replace the import block (lines 16-18):

```python
from ..cv.cell_extractor import extract_cells
from ..cv.detector import ChainDetector, warp_grid, DetectionResult
from ..cv.opencv_detector import OpenCVGridDetector
from ..cv.yolo_detector import YoloGridDetector
```

Replace the detector globals and functions (lines 30-95) with:

```python
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
    """Lazy-init the detector chain: YOLO (preferred) → OpenCV (fallback)."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR

    detectors = []

    yolo = YoloGridDetector(
        model_path=os.getenv("YOLO_MODEL_PATH"),
        num_keypoints=16,
    )
    if yolo.is_ready:
        detectors.append(yolo)
        _LOGGER.info("YOLO grid detector ready")
    else:
        _LOGGER.info("YOLO unavailable: %s", yolo.load_error)

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
```

- [ ] **Step 3: Update `solve_sudoku_from_image` to pass `detector_name`**

In the `solve_sudoku_from_image` endpoint (around line 266), change:

```python
        grid_img = detect_grid_image(img)
        if grid_img is None:
```

to:

```python
        det_result = detect_grid_image(img)
        if det_result is None:
            return _image_response(
                "Could not detect a Sudoku grid in the image. "
                "Please ensure the grid is clearly visible."
            )

        grid_img, detector_name = det_result
```

Then add `detector_name=detector_name` to all `_image_response(...)` calls within this endpoint. Update `_image_response` to accept and pass the new parameter:

In `_image_response` function, add `detector_name: str | None = None` parameter, and pass it to `ImageSolveResponse(...)`.

In `ocr_kwargs` dict, add `detector_name=detector_name`.

- [ ] **Step 4: Update `detect_grid` endpoint similarly**

In the `detect_grid` endpoint, change `detect_grid_image` call to unpack the tuple and include `detector_name` in the response dict.

- [ ] **Step 5: Run the web app test**

```bash
CNN_MODEL_PATH=models/releases/sudoku_digit_cnn_v3.3.1.onnx uv run python -c "
from backend.api.routes import detect_grid_image
import cv2
img = cv2.imread('data/raw/images/sudoku_10.jpeg')
result = detect_grid_image(img)
print(f'Result: warped={result[0].shape}, detector={result[1]}')
"
```
Expected: `Result: warped=(450, 450, 3), detector=yolo`

- [ ] **Step 6: Commit**

```bash
git add backend/models/schemas.py backend/api/routes.py
git commit -m "refactor: web app uses ChainDetector, returns detector_name in API"
```

---

### Task 5: Update Camera Monitor

**Files:**
- Modify: `camera_monitor/app.py`

- [ ] **Step 1: Update imports and detection call**

In `camera_monitor/app.py`, update the import (line 24):

```python
from backend.cv.yolo_detector import YoloGridDetector
from backend.cv.detector import warp_grid
```

In `_processing_loop` (around lines 85-93), change:

```python
            # YOLO detection: grid + 4 corner keypoints.
            det = detector.detect(frame)
            if det is None:
                LOGGER.debug("YOLO: no grid detected")
                result_holder.set(ProcessingResult())
                continue

            warped, corners, det_conf = det
            LOGGER.info("YOLO: conf=%.2f", det_conf)
```

to:

```python
            # YOLO detection.
            det = detector.detect(frame)
            if det is None:
                LOGGER.debug("YOLO: no grid detected")
                result_holder.set(ProcessingResult())
                continue

            corners = det.corners
            warped = warp_grid(frame, corners)
            LOGGER.info("YOLO: conf=%.2f detector=%s", detector.last_confidence, det.detector_name)
```

- [ ] **Step 2: Verify camera monitor module imports cleanly**

```bash
uv run python -c "from camera_monitor.app import CameraConfig; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add camera_monitor/app.py
git commit -m "refactor: camera monitor uses DetectionResult + warp_grid"
```

---

### Task 6: Update E2E Tests

**Files:**
- Modify: `tests/test_yolo_16kpt.py`

- [ ] **Step 1: Update `TestEndToEnd16Kpt._run_pipeline` for new `detect()` return type**

The `detect()` method now returns `DetectionResult` instead of `(warped, keypoints, conf)`. Update the test:

Find in `_run_pipeline`:
```python
        result = detector.detect(image)
        assert result is not None, f"YOLO detection failed for {img_path}"
        warped, keypoints, conf = result

        assert keypoints.shape == (16, 2), f"Expected 16 keypoints, got {keypoints.shape}"
```

Replace with:
```python
        from backend.cv.detector import warp_grid
        det = detector.detect(image)
        assert det is not None, f"YOLO detection failed for {img_path}"
        assert det.corners.shape == (4, 2), f"Expected 4 corners, got {det.corners.shape}"
        warped = warp_grid(image, det.corners)
```

Also remove `warped` and `keypoints` from any later references that no longer apply. The `cells = extract_cells_adaptive(image, keypoints)` line should be changed to use `extract_cells(warped)` since we're using old cell extraction for now:

```python
        cells = extract_cells(warped)
```

Update the import at the top of the file: replace `from backend.cv.adaptive_cell_extractor import extract_cells_adaptive` with `from backend.cv.cell_extractor import extract_cells` (it already imports `clean_cell` from there).

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_yolo_16kpt.py::TestParseKeypoints16 -v`
Expected: All 4 parsing tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_yolo_16kpt.py
git commit -m "test: update E2E tests for DetectionResult return type"
```
