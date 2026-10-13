"""CNN-based Sudoku digit reader using ONNX Runtime."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from ..cv.cell_extractor import clean_cell

DEFAULT_CNN_MODEL = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "releases"
    / "sudoku_digit_cnn.onnx"
)


class CnnDigitReader:
    """Read Sudoku digits with a self-trained CNN model exported to ONNX."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        blank_threshold: float = 0.65,
        digit_threshold: float = 0.55,
        top_k_candidates: int = 4,
    ):
        self.model_path = Path(model_path)
        self.blank_threshold = blank_threshold
        self.digit_threshold = digit_threshold
        self.top_k_candidates = top_k_candidates

        if not self.model_path.exists():
            raise FileNotFoundError(f"CNN model not found: {self.model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=sess_options, providers=["CPUExecutionProvider"],
        )
        input_meta = self._session.get_inputs()[0]
        self._input_name = input_meta.name
        self._output_name = self._session.get_outputs()[0].name
        self._input_size = int(input_meta.shape[2])  # NCHW

        self._model_version = self.model_path.stem
        self._class_values = list(range(10))
        meta_path = self.model_path.resolve().with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._model_version = meta.get("model_version") or meta.get("version") or self._model_version
            self._class_values = meta.get("class_values") or self._class_values

    @property
    def model_version(self) -> str:
        """Model version string."""
        return self._model_version

    def recognize_grid_with_metadata(
        self, cells: list[np.ndarray], threshold: float | None = None
    ) -> tuple[list[list[int]], dict[str, Any]]:
        """Recognize a full Sudoku grid and return OCR metadata."""
        digit_threshold = threshold if threshold is not None else self.digit_threshold

        start = time.perf_counter()

        # Phase 0: clean all cells and check blank likelihood (pure OpenCV).
        cleaned = [clean_cell(cell, cell_size=self._input_size) for cell in cells]
        likely_blanks = [self._is_likely_blank(cell) for cell in cleaned]

        # Phase 1: batch inference on all 81 cells.
        all_probs = self._predict_probs_batch(cleaned)

        cell_digits: list[int] = []
        cell_confidences: list[float] = []
        cell_candidates: list[list[tuple[int, float]]] = []
        recheck_indices: list[int] = []

        for idx in range(81):
            probs = all_probs[idx]
            candidates = self._top_digit_candidates(probs)
            value, conf = self._select_value(probs, digit_threshold)

            if 1 <= value <= 9 and conf >= digit_threshold:
                cell_digits.append(value)
                cell_confidences.append(conf)
                cell_candidates.append(candidates)
            elif value == 0 and conf >= self.blank_threshold and likely_blanks[idx]:
                cell_digits.append(0)
                cell_confidences.append(conf)
                cell_candidates.append(candidates)
            else:
                # Needs recheck — store primary result as placeholder.
                cell_digits.append(value)
                cell_confidences.append(conf)
                cell_candidates.append(candidates)
                recheck_indices.append(idx)

        # Phase 2: batch recheck for uncertain cells.
        if recheck_indices:
            recheck_images = [
                self._make_blank_recheck_variant(cleaned[i]) for i in recheck_indices
            ]
            recheck_probs = self._predict_probs_batch(recheck_images)

            for j, idx in enumerate(recheck_indices):
                probs_r = recheck_probs[j]
                cands_r = self._top_digit_candidates(probs_r)
                val_r, conf_r = self._select_value(probs_r, digit_threshold)

                primary_value = cell_digits[idx]
                primary_conf = cell_confidences[idx]

                if 1 <= val_r <= 9 and conf_r >= digit_threshold:
                    cell_digits[idx] = val_r
                    cell_confidences[idx] = conf_r
                    cell_candidates[idx] = cands_r
                elif 1 <= primary_value <= 9:
                    pass  # keep primary
                elif val_r == 0 and conf_r >= primary_conf:
                    cell_digits[idx] = 0
                    cell_confidences[idx] = conf_r
                    cell_candidates[idx] = cands_r

        grid = [cell_digits[i * 9 : (i + 1) * 9] for i in range(9)]

        recognized_confidences = [
            cell_confidences[i]
            for i, val in enumerate(cell_digits)
            if val != 0 and cell_confidences[i] > 0
        ]
        avg_confidence = (
            float(np.mean(recognized_confidences)) if recognized_confidences else None
        )

        latency_ms = (time.perf_counter() - start) * 1000.0
        flat_grid = [grid[i // 9][i % 9] for i in range(81)]
        cell_predictions = []
        for idx in range(81):
            value = int(flat_grid[idx])
            confidence = float(cell_confidences[idx])
            cell_predictions.append(
                {
                    "index": idx,
                    "row": idx // 9,
                    "col": idx % 9,
                    "value": value,
                    "confidence": confidence,
                    "candidates": [
                        [int(digit), float(prob)] for digit, prob in cell_candidates[idx]
                    ],
                }
            )

        metadata = {
            "average_confidence": avg_confidence,
            "latency_ms": latency_ms,
            "engine": "cnn",
            "model_version": self.model_version,
            "cell_predictions": cell_predictions,
        }
        return grid, metadata

    def _predict_probs_batch(self, images: list[np.ndarray]) -> np.ndarray:
        """Batch inference on cleaned gray cells: (N, num_classes) probabilities."""
        batch = np.stack(images).astype(np.float32)[:, None] / 255.0
        logits = self._session.run([self._output_name], {self._input_name: batch})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def _select_value(self, probs: np.ndarray, digit_threshold: float) -> tuple[int, float]:
        index = int(np.argmax(probs))
        conf = float(probs[index])
        value = self._class_values[index]
        if value != 0 and conf < digit_threshold:
            # Unknown digit is not a given; retain the actual blank probability.
            return 0, float(probs[self._class_values.index(0)])
        return value, conf

    def _top_digit_candidates(self, probs: np.ndarray) -> list[tuple[int, float]]:
        ranked = [
            (self._class_values[i], float(probs[i]))
            for i in np.argsort(probs)[::-1]
            if self._class_values[i] != 0
        ]
        return ranked[: self.top_k_candidates]

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
        small = cv2.resize(cell_image, (24, 24))
        if float(np.std(small)) < 4.0:
            return True
        _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return float(np.mean(binary > 0)) < 0.025
