"""CNN-based Sudoku digit reader using ONNX Runtime."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..cv.cell_extractor import clean_cell

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover - environment dependent
    ort = None


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "sudoku_digit_cnn_latest.onnx"
)


class CnnDigitReader:
    """Read Sudoku digits with a self-trained CNN model exported to ONNX."""

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        blank_threshold: float = 0.65,
        digit_threshold: float = 0.55,
        rerank_confidence: float = 0.80,
        top_k_candidates: int = 4,
        strict: bool = False,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.blank_threshold = blank_threshold
        self.digit_threshold = digit_threshold
        self.rerank_confidence = rerank_confidence
        self.top_k_candidates = max(1, top_k_candidates)
        self.strict = strict

        self._session = None
        self._input_name = ""
        self._output_name = ""
        self._input_size = 48
        self._nhwc_input = False
        self._class_values: list[int] = []
        self._model_version = self.model_path.stem
        self._load_error: Optional[str] = None

        self._load_model()

        if self.strict and not self.is_ready:
            raise RuntimeError(self._load_error or "CNN OCR model is not ready")

    @property
    def is_ready(self) -> bool:
        """Whether ONNX model session is ready."""
        return self._session is not None

    @property
    def load_error(self) -> Optional[str]:
        """Reason if model failed to load."""
        return self._load_error

    @property
    def model_version(self) -> str:
        """Model version string."""
        return self._model_version

    def recognize_grid(
        self, cells: list[np.ndarray], threshold: float | None = None
    ) -> list[list[int]]:
        """Recognize a 9x9 Sudoku grid from 81 cells."""
        grid, _ = self.recognize_grid_with_metadata(cells, threshold)
        return grid

    def recognize_grid_with_metadata(
        self, cells: list[np.ndarray], threshold: float | None = None
    ) -> tuple[list[list[int]], dict[str, Any]]:
        """Recognize a full Sudoku grid and return OCR metadata."""
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(cells)}")
        self._ensure_ready()

        digit_threshold = threshold if threshold is not None else self.digit_threshold

        start = time.perf_counter()
        cell_digits: list[int] = []
        cell_confidences: list[float] = []
        cell_candidates: list[list[tuple[int, float]]] = []

        for cell in cells:
            digit, confidence, candidates = self._predict_cell_with_candidates(
                cell, digit_threshold
            )
            cell_digits.append(digit)
            cell_confidences.append(confidence)
            cell_candidates.append(candidates)

        grid = [cell_digits[i * 9 : (i + 1) * 9] for i in range(9)]
        grid = self._rerank_with_constraints(grid, cell_confidences, cell_candidates)

        recognized_confidences = [
            cell_confidences[i]
            for i, val in enumerate(cell_digits)
            if val != 0 and cell_confidences[i] > 0
        ]
        avg_confidence = (
            float(np.mean(recognized_confidences)) if recognized_confidences else None
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        metadata = {
            "average_confidence": avg_confidence,
            "latency_ms": latency_ms,
            "engine": "cnn",
            "model_version": self.model_version,
        }
        return grid, metadata

    def predict_cell(self, cell_image: np.ndarray) -> tuple[Optional[int], float]:
        """Predict a single cell: return (digit or None, confidence)."""
        digit, confidence, _ = self._predict_cell_with_candidates(
            cell_image, self.digit_threshold
        )
        return (digit if digit != 0 else None), confidence

    def _load_model(self) -> None:
        if ort is None:
            self._load_error = (
                "onnxruntime is not installed. Install with: uv sync --extra ml"
            )
            return

        if not self.model_path.exists():
            self._load_error = (
                f"CNN model not found: {self.model_path}. "
                "Train and export model first (scripts/train_cnn_ocr.py + scripts/export_cnn_onnx.py)."
            )
            return

        try:
            providers = ["CPUExecutionProvider"]
            session = ort.InferenceSession(str(self.model_path), providers=providers)

            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if not inputs or not outputs:
                raise RuntimeError("Invalid ONNX model I/O metadata")

            input_meta = inputs[0]
            output_meta = outputs[0]
            self._input_name = input_meta.name
            self._output_name = output_meta.name
            self._session = session

            shape = input_meta.shape
            if len(shape) == 4:
                h, w = shape[2], shape[3]
                if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                    self._input_size = int(h)

                if shape[-1] == 1 and shape[1] != 1:
                    self._nhwc_input = True

            self._load_class_values(output_meta)
            self._load_metadata_sidecar()

        except Exception as exc:  # pragma: no cover - depends on runtime environment
            self._session = None
            self._load_error = f"Failed to load CNN OCR model: {exc}"

    def _load_class_values(self, output_meta: Any) -> None:
        output_shape = output_meta.shape
        class_count = None
        if len(output_shape) >= 2 and isinstance(output_shape[-1], int):
            class_count = output_shape[-1]

        if class_count == 10:
            self._class_values = list(range(10))
        elif class_count == 11:
            self._class_values = list(range(10)) + [-1]
        else:
            # Default map (0 blank + digits 1..9)
            self._class_values = list(range(10))

    def _load_metadata_sidecar(self) -> None:
        meta_path = self.model_path.with_suffix(".meta.json")
        if not meta_path.exists():
            return

        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            version = meta.get("version")
            if isinstance(version, str) and version.strip():
                self._model_version = version.strip()

            class_values = meta.get("class_values")
            if (
                isinstance(class_values, list)
                and class_values
                and all(isinstance(x, int) for x in class_values)
            ):
                self._class_values = class_values
        except Exception:
            # Sidecar is optional; ignore parse errors.
            return

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError(self._load_error or "CNN OCR model is not ready")

    def _predict_cell_with_candidates(
        self, cell_image: np.ndarray, digit_threshold: float
    ) -> tuple[int, float, list[tuple[int, float]]]:
        self._ensure_ready()

        if cell_image is None or cell_image.size == 0:
            return 0, 1.0, []

        normalized = clean_cell(cell_image, cell_size=self._input_size)
        likely_blank = self._is_likely_blank(normalized)

        probs_primary = self._predict_probs(normalized)
        candidates_primary = self._top_digit_candidates(probs_primary)
        primary_value, primary_conf = self._select_value(probs_primary)

        # Non-blank prediction with enough confidence.
        if 1 <= primary_value <= 9 and primary_conf >= digit_threshold:
            return primary_value, primary_conf, candidates_primary

        # Blank prediction with high confidence and blank-like appearance.
        if primary_value == 0 and primary_conf >= self.blank_threshold and likely_blank:
            return 0, primary_conf, candidates_primary

        # Blank recheck: reprocess once to recover thin anti-aliased digits.
        recheck = self._make_blank_recheck_variant(normalized)
        probs_recheck = self._predict_probs(recheck)
        candidates_recheck = self._top_digit_candidates(probs_recheck)
        recheck_value, recheck_conf = self._select_value(probs_recheck)

        if 1 <= recheck_value <= 9 and recheck_conf >= digit_threshold:
            return recheck_value, recheck_conf, candidates_recheck

        if 1 <= primary_value <= 9:
            return primary_value, primary_conf, candidates_primary

        # Keep blank when both attempts are inconclusive.
        if recheck_value == 0 and recheck_conf >= primary_conf:
            return 0, recheck_conf, candidates_recheck

        return 0, primary_conf, candidates_primary

    def _predict_probs(self, image: np.ndarray) -> np.ndarray:
        tensor = self._to_tensor(image)
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        logits = np.asarray(outputs[0], dtype=np.float32)

        if logits.ndim == 1:
            logits = logits[None, :]
        elif logits.ndim > 2:
            logits = logits.reshape(logits.shape[0], -1)

        return self._softmax(logits[0])

    def _to_tensor(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if image.shape[:2] != (self._input_size, self._input_size):
            image = cv2.resize(image, (self._input_size, self._input_size))

        image_f = image.astype(np.float32) / 255.0
        if self._nhwc_input:
            tensor = image_f[None, :, :, None]
        else:
            tensor = image_f[None, None, :, :]
        return tensor.astype(np.float32)

    def _select_value(self, probs: np.ndarray) -> tuple[int, float]:
        index = int(np.argmax(probs))
        conf = float(probs[index])

        if index >= len(self._class_values):
            return 0, conf

        value = self._class_values[index]
        if value == -1:
            return 0, conf
        return value, conf

    def _top_digit_candidates(self, probs: np.ndarray) -> list[tuple[int, float]]:
        ranked_indices = np.argsort(probs)[::-1]
        out: list[tuple[int, float]] = []

        for idx in ranked_indices:
            i = int(idx)
            if i >= len(self._class_values):
                continue
            value = self._class_values[i]
            if not (1 <= value <= 9):
                continue
            out.append((value, float(probs[i])))
            if len(out) >= self.top_k_candidates:
                break

        return out

    def _rerank_with_constraints(
        self,
        grid: list[list[int]],
        confidences: list[float],
        candidates: list[list[tuple[int, float]]],
    ) -> list[list[int]]:
        """Apply lightweight Sudoku-constraint rerank on uncertain cells."""
        reranked = [row[:] for row in grid]

        # Drop obviously invalid low-confidence digits first.
        for row in range(9):
            for col in range(9):
                idx = row * 9 + col
                val = reranked[row][col]
                if val == 0:
                    continue
                if confidences[idx] >= self.rerank_confidence:
                    continue
                if not self._is_valid_placement(reranked, row, col, val):
                    reranked[row][col] = 0

        unresolved = []
        for row in range(9):
            for col in range(9):
                idx = row * 9 + col
                if reranked[row][col] == 0 or confidences[idx] < self.rerank_confidence:
                    unresolved.append((row, col, idx))

        unresolved.sort(key=lambda item: confidences[item[2]])

        for _ in range(2):
            changed = False
            for row, col, idx in unresolved:
                current = reranked[row][col]

                if current != 0 and self._is_valid_placement(reranked, row, col, current):
                    continue

                for digit, prob in candidates[idx]:
                    if prob < 0.15:
                        continue
                    if self._is_valid_placement(reranked, row, col, digit):
                        if current != digit:
                            reranked[row][col] = digit
                            changed = True
                        break
            if not changed:
                break

        return reranked

    def _is_valid_placement(
        self, grid: list[list[int]], row: int, col: int, val: int
    ) -> bool:
        if val == 0:
            return True

        # Row check
        for c in range(9):
            if c != col and grid[row][c] == val:
                return False

        # Col check
        for r in range(9):
            if r != row and grid[r][col] == val:
                return False

        # 3x3 check
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if (r != row or c != col) and grid[r][c] == val:
                    return False

        return True

    def _make_blank_recheck_variant(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2
        )

        # Keep white background, black foreground.
        if np.mean(adaptive) < 127:
            adaptive = cv2.bitwise_not(adaptive)

        kernel = np.ones((2, 2), np.uint8)
        return cv2.erode(adaptive, kernel, iterations=1)

    def _is_likely_blank(self, cell_image: np.ndarray) -> bool:
        if cell_image is None or cell_image.size == 0:
            return True

        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image

        small = cv2.resize(gray, (24, 24)) if gray.shape[:2] != (24, 24) else gray

        if float(np.std(small)) < 4.0:
            return True

        _, binary = cv2.threshold(
            small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        ink_ratio = float(np.mean(binary > 0))
        return ink_ratio < 0.015

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
