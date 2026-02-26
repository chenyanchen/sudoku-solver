"""Realtime macOS screen monitor with Sudoku overlay hints."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import signal
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.api.routes import prepare_grid_for_ocr
from backend.cv.cell_extractor import extract_cells
from backend.cv.grid_detector import find_grids_with_corners
from backend.ocr.cnn_digit_reader import CnnDigitReader
from backend.ocr.grid_repair import try_repair_grid_with_candidates
from backend.solver.backtracking import SudokuSolver

LOGGER = logging.getLogger("screen_monitor")


@dataclass
class MonitorConfig:
    model_path: Optional[str] = None
    monitor_index: int = 0
    idle_fps: float = 2.0
    active_fps: float = 6.0
    active_seconds: float = 2.0
    stable_frames: int = 2
    lost_frames: int = 3
    frame_change_threshold: float = 1.8
    bbox_iou_threshold: float = 0.85
    grid_size_delta_ratio: float = 0.08
    min_givens: int = 17
    min_bbox_area_ratio: float = 0.03
    overlay_hold_seconds: float = 2.0
    solve_cache_max: int = 20
    detect_cache_max: int = 64
    quantize_step: int = 4
    debug: bool = False


@dataclass
class StabilityTracker:
    stable_count: int = 0
    lost_count: int = 0
    last_signature: Optional[str] = None
    last_bbox: Optional[tuple[int, int, int, int]] = None

    def on_no_grid(self) -> int:
        self.lost_count += 1
        self.stable_count = 0
        return self.lost_count

    def on_grid(
        self,
        signature: str,
        bbox: tuple[int, int, int, int],
        cfg: MonitorConfig,
    ) -> int:
        if grids_match(
            self.last_signature,
            self.last_bbox,
            signature,
            bbox,
            cfg.bbox_iou_threshold,
            cfg.grid_size_delta_ratio,
        ):
            self.stable_count += 1
        else:
            self.stable_count = 1
        self.lost_count = 0
        self.last_signature = signature
        self.last_bbox = bbox
        return self.stable_count


class LruCache:
    """Small LRU cache for frame/detection/solve stages."""

    def __init__(self, max_size: int):
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)


def make_thumbnail(frame_bgr: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def mean_abs_diff(img_a: np.ndarray, img_b: np.ndarray) -> float:
    diff = cv2.absdiff(img_a, img_b)
    return float(np.mean(diff))


def is_frame_changed(
    previous_thumb: Optional[np.ndarray],
    current_thumb: np.ndarray,
    threshold: float,
) -> bool:
    if previous_thumb is None:
        return True
    return mean_abs_diff(previous_thumb, current_thumb) > float(threshold)


def thumbnail_hash(thumb_gray: np.ndarray) -> str:
    return hashlib.sha1(thumb_gray.tobytes()).hexdigest()


def grid_signature(warped_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    normalized = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    return hashlib.sha1(normalized.tobytes()).hexdigest()


def puzzle_signature(grid: list[list[int]]) -> str:
    flat = "".join(str(int(cell)) for row in grid for cell in row)
    return hashlib.sha1(flat.encode("utf-8")).hexdigest()


def bbox_from_corners(corners: np.ndarray) -> tuple[int, int, int, int]:
    xs = corners[:, 0]
    ys = corners[:, 1]
    left = int(np.min(xs))
    top = int(np.min(ys))
    right = int(np.max(xs))
    bottom = int(np.max(ys))
    return left, top, right, bottom


def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(area_a + area_b - inter_area)


def size_delta_ratio(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return abs(area_a - area_b) / float(max(area_a, area_b))


def grids_match(
    previous_signature: Optional[str],
    previous_bbox: Optional[tuple[int, int, int, int]],
    current_signature: str,
    current_bbox: tuple[int, int, int, int],
    iou_threshold: float,
    area_delta_ratio_threshold: float,
) -> bool:
    if previous_signature == current_signature and previous_signature is not None:
        return True
    if previous_bbox is None:
        return False
    if bbox_iou(previous_bbox, current_bbox) < iou_threshold:
        return False
    return size_delta_ratio(previous_bbox, current_bbox) <= area_delta_ratio_threshold


def quantize_bbox(
    bbox: tuple[int, int, int, int],
    step: int,
) -> tuple[int, int, int, int]:
    q = max(1, int(step))
    return tuple(int(round(v / q) * q) for v in bbox)


def build_render_key(
    signature: str,
    bbox: tuple[int, int, int, int],
    dpr: float,
    quantize_step: int,
) -> tuple[str, tuple[int, int, int, int], float]:
    return (signature, quantize_bbox(bbox, quantize_step), round(float(dpr), 3))


def should_render(last_key: Any, new_key: Any) -> bool:
    return last_key != new_key


def _detect_candidates_raw(frame_bgr: np.ndarray, max_grids: int) -> list[tuple[np.ndarray, np.ndarray]]:
    ratios = (0.08, 0.04, 0.02)
    scan_limit = max(1, int(max_grids))

    for ratio in ratios:
        candidates = find_grids_with_corners(
            frame_bgr,
            max_grids=scan_limit,
            min_area_ratio=ratio,
        )
        if candidates:
            return _dedupe_candidates(candidates, max_candidates=scan_limit)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    for ratio in ratios:
        candidates = find_grids_with_corners(
            enhanced,
            max_grids=scan_limit,
            min_area_ratio=ratio,
        )
        if candidates:
            return _dedupe_candidates(candidates, max_candidates=scan_limit)

    return []


def _dedupe_candidates(
    candidates: list[tuple[np.ndarray, np.ndarray]],
    max_candidates: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    seen: set[tuple[int, int, int, int]] = set()

    for warped, corners in candidates:
        bbox = bbox_from_corners(corners)
        key = tuple(int(round(v / 12.0) * 12) for v in bbox)
        if key in seen:
            continue
        seen.add(key)
        unique.append((warped, corners))
        if len(unique) >= max_candidates:
            break

    return unique


def detect_candidates_with_corners(frame_bgr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return _detect_candidates_raw(frame_bgr, max_grids=8)


def solve_hints_from_warped(
    warped_bgr: np.ndarray,
    reader: CnnDigitReader,
    min_givens: int,
) -> tuple[Optional[dict[str, Any]], str, int]:
    ocr_grid = prepare_grid_for_ocr(warped_bgr)
    cells = extract_cells(ocr_grid)
    if len(cells) != 81:
        return None, "bad-cells", 0

    original_grid, metadata = reader.recognize_grid_with_metadata(cells, threshold=None)
    given_cells = sum(1 for row in original_grid for cell in row if cell != 0)
    if given_cells < min_givens:
        return None, f"low-givens:{given_cells}", given_cells

    solver = SudokuSolver()
    solved_grid = solver.solve(original_grid)

    if solved_grid is None:
        predictions = metadata.get("cell_predictions")
        repaired_grid, repaired_solution, _ = try_repair_grid_with_candidates(
            grid=original_grid,
            cell_predictions=predictions if isinstance(predictions, list) else None,
            max_changes=4,
            max_cells=20,
        )
        if repaired_solution is not None:
            original_grid = repaired_grid
            solved_grid = repaired_solution

    if solved_grid is None:
        return None, f"unsolved:{given_cells}", given_cells

    hints = []
    for row in range(9):
        for col in range(9):
            if original_grid[row][col] == 0:
                hints.append({"row": row, "col": col, "digit": int(solved_grid[row][col])})

    if not hints:
        return None, f"no-hints:{given_cells}", given_cells

    return {
        "original_grid": original_grid,
        "solved_grid": solved_grid,
        "hints": hints,
        "givens": given_cells,
        "metadata": metadata,
    }, "ok", given_cells


def compute_hint_positions(
    corners: np.ndarray,
    hints: list[dict[str, Any]],
    logical_offset: tuple[float, float] = (0.0, 0.0),
    capture_to_logical_scale: tuple[float, float] = (1.0, 1.0),
) -> list[dict[str, Any]]:
    if not hints:
        return []

    dst = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
    matrix = cv2.getPerspectiveTransform(dst, corners.astype(np.float32))

    source_points = []
    for item in hints:
        row = int(item["row"])
        col = int(item["col"])
        source_points.append([(col + 0.5) * 50.0, (row + 0.5) * 50.0])

    points = np.float32(source_points).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    logical_x0 = float(logical_offset[0])
    logical_y0 = float(logical_offset[1])
    scale_x = max(float(capture_to_logical_scale[0]), 1e-6)
    scale_y = max(float(capture_to_logical_scale[1]), 1e-6)
    output = []
    for i, item in enumerate(hints):
        output.append(
            {
                "x": float(logical_x0 + transformed[i, 0] * scale_x),
                "y": float(logical_y0 + transformed[i, 1] * scale_y),
                "digit": int(item["digit"]),
            }
        )
    return output


def _configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def _parse_args() -> MonitorConfig:
    parser = argparse.ArgumentParser(description="Realtime Sudoku screen monitor (macOS)")
    parser.add_argument(
        "--model",
        default=os.getenv("CNN_MODEL_PATH"),
        help="CNN model path (fallback to CNN_MODEL_PATH env or default model)",
    )
    parser.add_argument("--idle-fps", type=float, default=2.0)
    parser.add_argument("--active-fps", type=float, default=6.0)
    parser.add_argument("--active-seconds", type=float, default=2.0)
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument("--lost-frames", type=int, default=3)
    parser.add_argument("--min-bbox-area-ratio", type=float, default=0.03)
    parser.add_argument("--overlay-hold-seconds", type=float, default=2.0)
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=int(os.getenv("SCREEN_MONITOR_INDEX", "0")),
        help="MSS monitor index: 0 means all monitors, 1/2/... means specific monitor",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    return MonitorConfig(
        model_path=args.model,
        monitor_index=args.monitor_index,
        idle_fps=args.idle_fps,
        active_fps=args.active_fps,
        active_seconds=args.active_seconds,
        stable_frames=args.stable_frames,
        lost_frames=args.lost_frames,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        overlay_hold_seconds=args.overlay_hold_seconds,
        debug=args.debug,
    )


def run_monitor(config: MonitorConfig) -> int:
    _configure_logging(config.debug)

    try:
        import mss
        from PyQt6.QtCore import QObject, QPoint, Qt, pyqtSignal
        from PyQt6.QtGui import QColor, QFont, QPainter
        from PyQt6.QtWidgets import QApplication, QWidget
    except ImportError as exc:
        LOGGER.error("Missing screen dependencies. Run: uv sync --extra screen")
        LOGGER.error("Import error: %s", exc)
        return 2

    class OverlaySignals(QObject):
        update_signal = pyqtSignal(list)
        clear_signal = pyqtSignal()

    class OverlayWindow(QWidget):
        def __init__(self):
            super().__init__()
            self._cells: list[dict[str, Any]] = []
            window_flags = (
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.WindowTransparentForInput
            )
            # macOS 下 Tool 窗口会在应用失焦时隐藏，不能用于跨应用覆盖层。
            if hasattr(Qt.WindowType, "WindowDoesNotAcceptFocus"):
                window_flags |= Qt.WindowType.WindowDoesNotAcceptFocus
            self.setWindowFlags(
                window_flags
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            if hasattr(Qt.WidgetAttribute, "WA_ShowWithoutActivating"):
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            screen = QApplication.primaryScreen()
            if screen is not None:
                self.setGeometry(screen.virtualGeometry())

        def set_cells(self, cells: list[dict[str, Any]]) -> None:
            self._cells = cells
            self.update()

        def clear_cells(self) -> None:
            if not self._cells:
                return
            self._cells = []
            self.update()

        def paintEvent(self, _event) -> None:  # pragma: no cover - GUI runtime path
            if not self._cells:
                return

            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setFont(QFont("Helvetica Neue", 20, QFont.Weight.Bold))
            for cell in self._cells:
                x = int(cell["x"])
                y = int(cell["y"])
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(30, 80, 200, 150))
                painter.drawEllipse(QPoint(x, y), 13, 13)
                painter.setPen(QColor(255, 255, 255, 240))
                painter.drawText(x - 7, y + 7, str(cell["digit"]))

    app = QApplication.instance() or QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()
    overlay.raise_()

    signals = OverlaySignals()
    signals.update_signal.connect(overlay.set_cells)
    signals.clear_signal.connect(overlay.clear_cells)

    reader = CnnDigitReader(model_path=config.model_path, strict=False)
    if not reader.is_ready:
        raise RuntimeError(reader.load_error or "CNN model is not ready")

    stop_event = threading.Event()
    app.aboutToQuit.connect(stop_event.set)

    primary_screen = app.primaryScreen()
    screens = app.screens()

    def capture_worker() -> None:  # pragma: no cover - threaded runtime path
        try:
            tracker = StabilityTracker()
            detect_cache = LruCache(config.detect_cache_max)
            solve_cache = LruCache(config.solve_cache_max)

            overlay_visible = False
            last_render_key: Any = None
            last_valid_solution_at = 0.0
            prev_thumbs: dict[int, np.ndarray] = {}
            active_until = 0.0

            with mss.mss() as sct:
                monitors = sct.monitors
                available_monitor_count = max(0, len(monitors) - 1)
                if available_monitor_count <= 0:
                    raise RuntimeError("No screen monitors available from mss")

                if config.monitor_index == 0:
                    monitor_ids = list(range(1, len(monitors)))
                else:
                    monitor_ids = [int(config.monitor_index)]

                for monitor_id in monitor_ids:
                    if monitor_id < 1 or monitor_id >= len(monitors):
                        raise RuntimeError(
                            f"Invalid monitor index {monitor_id}, available range: 1..{len(monitors)-1}"
                        )

                monitor_meta: dict[int, dict[str, Any]] = {}
                for monitor_id in monitor_ids:
                    monitor = monitors[monitor_id]
                    if 1 <= monitor_id <= len(screens):
                        screen = screens[monitor_id - 1]
                    else:
                        screen = primary_screen
                    if screen is not None:
                        geometry = screen.geometry()
                        logical_offset = (float(geometry.x()), float(geometry.y()))
                        monitor_w = max(1.0, float(monitor["width"]))
                        monitor_h = max(1.0, float(monitor["height"]))
                        scale_x = float(geometry.width()) / monitor_w
                        scale_y = float(geometry.height()) / monitor_h
                    else:
                        logical_offset = (0.0, 0.0)
                        scale_x = 1.0
                        scale_y = 1.0

                    monitor_meta[monitor_id] = {
                        "monitor": monitor,
                        "logical_offset": logical_offset,
                        "capture_to_logical_scale": (scale_x, scale_y),
                    }

                # Permission probe per monitor: fail fast with a clear message.
                for meta in monitor_meta.values():
                    _ = np.array(sct.grab(meta["monitor"]))

                LOGGER.info(
                    "screen monitor started: monitor_index=%s active_monitors=%s",
                    config.monitor_index,
                    monitor_ids,
                )

                while not stop_event.is_set():
                    loop_start = time.perf_counter()
                    now = time.monotonic()

                    frames: list[tuple[int, np.ndarray, np.ndarray, bool]] = []
                    any_changed = False
                    for monitor_id in monitor_ids:
                        monitor = monitor_meta[monitor_id]["monitor"]
                        frame_raw = np.array(sct.grab(monitor), dtype=np.uint8)
                        frame_bgr = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2BGR)
                        thumb = make_thumbnail(frame_bgr)
                        changed = is_frame_changed(
                            prev_thumbs.get(monitor_id),
                            thumb,
                            config.frame_change_threshold,
                        )
                        prev_thumbs[monitor_id] = thumb
                        any_changed = any_changed or changed
                        frames.append((monitor_id, frame_bgr, thumb, changed))

                    if any_changed:
                        active_until = max(active_until, now + config.active_seconds)
                    elif overlay_visible:
                        target_fps = (
                            config.active_fps if now < active_until else config.idle_fps
                        )
                        _sleep_until_next(loop_start, target_fps, stop_event)
                        continue

                    selected_corners: Optional[np.ndarray] = None
                    selected_signature: Optional[str] = None
                    selected_bbox: Optional[tuple[int, int, int, int]] = None
                    solved_payload: Optional[dict[str, Any]] = None
                    selected_monitor_id: Optional[int] = None
                    selected_render_scale: float = 1.0
                    selected_logical_offset = (0.0, 0.0)
                    selected_capture_to_logical_scale = (1.0, 1.0)
                    best_givens = -1

                    for monitor_id, frame_bgr, thumb, changed in frames:
                        if not changed and overlay_visible:
                            continue

                        frame_key = f"m{monitor_id}:{thumbnail_hash(thumb)}"
                        candidates = detect_cache.get(frame_key)
                        if candidates is None:
                            candidates = detect_candidates_with_corners(frame_bgr)
                            detect_cache.put(frame_key, candidates)

                        if config.debug and candidates:
                            LOGGER.debug(
                                "monitor %d candidate grids detected: %d",
                                monitor_id,
                                len(candidates),
                            )

                        if not candidates:
                            continue

                        candidate_count = len(candidates)
                        for idx, (warped, corners) in enumerate(candidates):
                            bbox = bbox_from_corners(corners)
                            bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                            frame_area = max(1, frame_bgr.shape[0] * frame_bgr.shape[1])
                            bbox_ratio = bbox_area / float(frame_area)
                            if bbox_ratio < config.min_bbox_area_ratio:
                                if config.debug:
                                    LOGGER.debug(
                                        "monitor %d candidate rejected (%d/%d): tiny-bbox:%.4f",
                                        monitor_id,
                                        idx + 1,
                                        candidate_count,
                                        bbox_ratio,
                                    )
                                continue

                            raw_signature = grid_signature(warped)
                            cached_result = solve_cache.get(raw_signature)
                            if cached_result is None:
                                cached_payload, reason, givens = solve_hints_from_warped(
                                    warped,
                                    reader,
                                    config.min_givens,
                                )
                                solve_cache.put(
                                    raw_signature,
                                    {
                                        "payload": cached_payload,
                                        "reason": reason,
                                        "givens": int(givens),
                                    },
                                )
                            else:
                                cached_payload = cached_result.get("payload")
                                reason = str(cached_result.get("reason", "cache-hit"))
                                givens = int(cached_result.get("givens") or 0)

                            if cached_payload is None:
                                if config.debug:
                                    LOGGER.debug(
                                        "monitor %d candidate rejected (%d/%d): %s",
                                        monitor_id,
                                        idx + 1,
                                        candidate_count,
                                        reason,
                                    )
                                continue

                            givens = int(cached_payload.get("givens") or givens or 0)
                            if givens <= best_givens:
                                continue

                            meta = monitor_meta[monitor_id]
                            selected_monitor_id = monitor_id
                            selected_logical_offset = tuple(meta["logical_offset"])
                            selected_capture_to_logical_scale = tuple(
                                meta["capture_to_logical_scale"]
                            )
                            selected_render_scale = float(
                                (selected_capture_to_logical_scale[0]
                                 + selected_capture_to_logical_scale[1])
                                / 2.0
                            )
                            selected_corners = corners
                            puzzle_sig = puzzle_signature(cached_payload["original_grid"])
                            selected_signature = f"m{monitor_id}:g:{puzzle_sig}"
                            selected_bbox = bbox
                            solved_payload = cached_payload
                            best_givens = givens

                            if config.debug:
                                LOGGER.debug(
                                    "monitor %d candidate selected (%d/%d): givens=%s",
                                    monitor_id,
                                    idx + 1,
                                    candidate_count,
                                    givens,
                                )

                    if (
                        selected_monitor_id is None
                        or selected_corners is None
                        or selected_signature is None
                        or selected_bbox is None
                        or solved_payload is None
                    ):
                        lost = tracker.on_no_grid()
                        should_clear = (
                            overlay_visible
                            and lost >= config.lost_frames
                            and (now - last_valid_solution_at) >= config.overlay_hold_seconds
                        )
                        if should_clear:
                            signals.clear_signal.emit()
                            if config.debug:
                                LOGGER.debug(
                                    "overlay cleared: no valid grid for %d frames (hold %.2fs elapsed)",
                                    lost,
                                    now - last_valid_solution_at,
                                )
                            overlay_visible = False
                            last_render_key = None
                        target_fps = (
                            config.active_fps if now < active_until else config.idle_fps
                        )
                        _sleep_until_next(loop_start, target_fps, stop_event)
                        continue

                    stable = tracker.on_grid(selected_signature, selected_bbox, config)
                    if stable < config.stable_frames:
                        target_fps = (
                            config.active_fps if now < active_until else config.idle_fps
                        )
                        _sleep_until_next(loop_start, target_fps, stop_event)
                        continue

                    hint_cells = compute_hint_positions(
                        selected_corners,
                        solved_payload["hints"],
                        selected_logical_offset,
                        selected_capture_to_logical_scale,
                    )
                    if not hint_cells:
                        if overlay_visible:
                            signals.clear_signal.emit()
                            if config.debug:
                                LOGGER.debug("overlay cleared: hint list empty")
                            overlay_visible = False
                            last_render_key = None
                        target_fps = (
                            config.active_fps if now < active_until else config.idle_fps
                        )
                        _sleep_until_next(loop_start, target_fps, stop_event)
                        continue

                    last_valid_solution_at = now
                    render_key = build_render_key(
                        selected_signature,
                        selected_bbox,
                        selected_render_scale,
                        config.quantize_step,
                    )
                    if should_render(last_render_key, render_key):
                        signals.update_signal.emit(hint_cells)
                        if config.debug:
                            LOGGER.debug(
                                "overlay updated: monitor=%s hints=%d stable=%d",
                                selected_monitor_id,
                                len(hint_cells),
                                stable,
                            )
                        last_render_key = render_key
                        overlay_visible = True

                    target_fps = (
                        config.active_fps if now < active_until else config.idle_fps
                    )
                    _sleep_until_next(loop_start, target_fps, stop_event)
        except Exception:
            LOGGER.exception("capture worker crashed")
            stop_event.set()

    worker = threading.Thread(target=capture_worker, daemon=True)
    worker.start()

    def _handle_signal(_sig: int, _frame: Any) -> None:
        stop_event.set()
        app.quit()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    exit_code = app.exec()
    stop_event.set()
    worker.join(timeout=2.0)
    return int(exit_code)


def _sleep_until_next(loop_start: float, target_fps: float, stop_event: threading.Event) -> None:
    interval = 1.0 / max(float(target_fps), 0.1)
    elapsed = time.perf_counter() - loop_start
    remaining = max(0.0, interval - elapsed)
    stop_event.wait(remaining)


def main() -> int:
    config = _parse_args()
    return run_monitor(config)


if __name__ == "__main__":
    raise SystemExit(main())
