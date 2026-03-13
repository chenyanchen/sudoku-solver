"""Camera monitor application: capture, detect, solve, display."""

from __future__ import annotations

import logging
import queue
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from backend.ocr.cnn_digit_reader import CnnDigitReader
from screen_monitor.renderer import compute_hint_positions
from screen_monitor.solver_pipeline import ocr_grid_from_warped, solve_from_grid

from . import ProcessingResult
from .widget import CameraWidget
from .yolo_detector import YoloGridDetector

LOGGER = logging.getLogger("camera_monitor")


def _prepare_grid_for_camera_ocr(grid_img: np.ndarray) -> np.ndarray:
    """Camera-optimized preprocessing: apply CLAHE for uneven lighting."""
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY) if len(grid_img.shape) == 3 else grid_img
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))
    enhanced = clahe.apply(gray)
    if np.mean(enhanced) <= 128:
        enhanced = cv2.bitwise_not(enhanced)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


@dataclass
class CameraConfig:
    camera_index: int = 0
    model_path: Optional[str] = None
    yolo_model_path: Optional[str] = None
    fps: int = 30
    min_givens: int = 17
    min_confidence: float = 0.30
    yolo_conf: float = 0.3
    debug: bool = False


class ResultHolder:
    """Thread-safe container for the latest processing result."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._result = ProcessingResult()
        self._version = 0

    def get(self) -> tuple[ProcessingResult, int]:
        with self._lock:
            return self._result, self._version

    def set(self, result: ProcessingResult) -> None:
        with self._lock:
            self._version += 1
            self._result = result


def _processing_loop(
    frame_queue: queue.Queue,
    result_holder: ResultHolder,
    reader: CnnDigitReader,
    detector: YoloGridDetector,
    config: CameraConfig,
    stop_event: threading.Event,
) -> None:
    """Background thread: YOLO detection + OCR + solve."""
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            # YOLO detection: grid + 4 corner keypoints.
            det = detector.detect(frame)
            if det is None:
                LOGGER.debug("YOLO: no grid detected")
                result_holder.set(ProcessingResult())
                continue

            warped, corners, det_conf = det
            LOGGER.info("YOLO: conf=%.2f", det_conf)

            # OCR (camera needs CLAHE for uneven lighting).
            ocr_result, reason, givens = ocr_grid_from_warped(
                warped, reader, config.min_givens,
                prepare_fn=_prepare_grid_for_camera_ocr,
            )
            if ocr_result is None:
                LOGGER.info("OCR failed: reason=%s givens=%d", reason, givens)
                result_holder.set(ProcessingResult(
                    corners=corners, warped=warped, givens=givens,
                ))
                continue

            # Gate: reject low-confidence OCR results.
            avg_conf = ocr_result.get("metadata", {}).get("average_confidence")
            if avg_conf is not None and avg_conf < config.min_confidence:
                LOGGER.info("OCR rejected: avg_conf=%.2f < %.2f",
                            avg_conf, config.min_confidence)
                result_holder.set(ProcessingResult(
                    corners=corners, warped=warped, givens=givens,
                ))
                continue

            # Solve.
            payload, solve_reason = solve_from_grid(ocr_result)
            if payload is None:
                LOGGER.info("Solve failed: reason=%s givens=%d", solve_reason, givens)
                result_holder.set(ProcessingResult(
                    corners=corners, warped=warped,
                    original_grid=ocr_result["original_grid"],
                    givens=givens,
                ))
                continue

            hints = payload["hints"]
            positioned = compute_hint_positions(
                corners, hints, (0.0, 0.0), (1.0, 1.0),
            )

            result_holder.set(ProcessingResult(
                corners=corners,
                warped=warped,
                original_grid=payload["original_grid"],
                solved_grid=payload["solved_grid"],
                hints=positioned,
                givens=givens,
            ))

        except Exception:
            LOGGER.warning("processing error", exc_info=True)
            result_holder.set(ProcessingResult())


def run_camera(config: CameraConfig) -> int:
    """Main entry point: open camera, launch Qt app, run processing loop."""
    app = QApplication(sys.argv)

    # Open camera.
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        LOGGER.error("Cannot open camera index %d", config.camera_index)
        return 1
    LOGGER.info("Camera %d opened: %.0fx%.0f",
                config.camera_index,
                cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load YOLO grid detector.
    detector = YoloGridDetector(
        model_path=config.yolo_model_path,
        conf_threshold=config.yolo_conf,
    )
    if not detector.is_ready:
        LOGGER.error("YOLO model failed to load: %s", detector.load_error)
        cap.release()
        return 1

    # Load OCR model.
    reader = CnnDigitReader(model_path=config.model_path)
    if not reader.is_ready:
        LOGGER.error("CNN model failed to load: %s", reader.load_error)
        cap.release()
        return 1
    LOGGER.info("CNN model loaded: %s", reader.model_version)

    widget = CameraWidget(debug=config.debug)
    widget.show()

    result_holder = ResultHolder()
    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()

    worker = threading.Thread(
        target=_processing_loop,
        args=(frame_queue, result_holder, reader, detector, config, stop_event),
        daemon=True,
    )
    worker.start()

    last_version = -1

    def on_timer() -> None:
        nonlocal last_version
        ret, frame = cap.read()
        if not ret or frame is None:
            return

        widget.set_frame(frame)

        # Feed latest frame to processing thread (drop old if busy).
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put(frame)

        # Apply latest result to widget (only when changed).
        result, version = result_holder.get()
        if version != last_version:
            last_version = version
            widget.set_result(result)

    timer = QTimer()
    timer.timeout.connect(on_timer)
    timer.start(1000 // config.fps)

    # Graceful shutdown on SIGINT/SIGTERM.
    def _signal_handler(*_: Any) -> None:
        app.quit()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    exit_code = app.exec()

    # Cleanup.
    stop_event.set()
    timer.stop()
    cap.release()
    worker.join(timeout=2.0)
    return exit_code
