"""Main coordination: capture loop, state machine, and Qt app lifecycle."""

from __future__ import annotations

import logging
import signal
import sys
import threading
import time
from typing import Any, Optional

import numpy as np

from backend.ocr.cnn_digit_reader import CnnDigitReader

from .detector import detect_candidates_with_corners
from .frame_hasher import is_frame_changed, make_thumbnail, thumbnail_hash
from .grid_tracker import (
    StabilityTracker,
    bbox_from_corners,
    grid_signature,
    puzzle_signature,
)
from .renderer import build_render_key, compute_hint_positions, should_render
from .solver_pipeline import solve_hints_from_warped
from .types import LruCache, MonitorConfig

LOGGER = logging.getLogger("screen_monitor")


def configure_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def run_monitor(config: MonitorConfig) -> int:
    configure_logging(config.debug)

    try:
        import mss
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        LOGGER.error("Missing screen dependencies. Run: uv sync --extra screen")
        LOGGER.error("Import error: %s", exc)
        return 2

    from .overlay import OverlaySignals, OverlayWindow

    app = QApplication.instance() or QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()
    overlay.raise_()

    signals = OverlaySignals()
    signals.update_signal.connect(overlay.set_cells)
    signals.clear_signal.connect(overlay.clear_cells)
    # FIX #2: worker crash triggers app exit instead of zombie overlay.
    signals.error_signal.connect(app.quit)

    reader = CnnDigitReader(model_path=config.model_path, strict=False)
    if not reader.is_ready:
        LOGGER.critical("CNN model failed to load: %s", reader.load_error)
        return 1

    stop_event = threading.Event()
    app.aboutToQuit.connect(stop_event.set)

    primary_screen = app.primaryScreen()
    screens = app.screens()

    def capture_worker() -> None:  # pragma: no cover
        try:
            _capture_loop(
                config, reader, signals, stop_event,
                mss, screens, primary_screen,
            )
        except Exception:
            LOGGER.exception("capture worker crashed")
            # FIX #2: notify Qt to quit.
            signals.error_signal.emit()
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


def _capture_loop(
    config: MonitorConfig,
    reader: CnnDigitReader,
    signals: Any,
    stop_event: threading.Event,
    mss: Any,
    screens: list,
    primary_screen: Any,
) -> None:
    """Main capture → detect → solve → render loop (runs in daemon thread)."""
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
        if len(monitors) <= 1:
            raise RuntimeError("No screen monitors available from mss")

        if config.monitor_index == 0:
            monitor_ids = list(range(1, len(monitors)))
        else:
            monitor_ids = [int(config.monitor_index)]

        for mid in monitor_ids:
            if mid < 1 or mid >= len(monitors):
                raise RuntimeError(
                    f"Invalid monitor index {mid}, available: 1..{len(monitors)-1}"
                )

        monitor_meta: dict[int, dict[str, Any]] = {}
        for mid in monitor_ids:
            monitor = monitors[mid]
            if 1 <= mid <= len(screens):
                screen = screens[mid - 1]
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
                scale_x, scale_y = 1.0, 1.0

            monitor_meta[mid] = {
                "monitor": monitor,
                "logical_offset": logical_offset,
                "capture_to_logical_scale": (scale_x, scale_y),
            }

        # FIX #4: permission probe — check for all-black frames.
        for mid, meta in monitor_meta.items():
            probe = np.array(sct.grab(meta["monitor"]), dtype=np.uint8)
            if probe[:, :, :3].max() == 0:
                raise RuntimeError(
                    "Screen capture returned a blank frame (monitor %d). "
                    "Grant Screen Recording permission: "
                    "System Settings → Privacy & Security → Screen Recording" % mid
                )

        LOGGER.info(
            "screen monitor started: monitor_index=%s active_monitors=%s",
            config.monitor_index, monitor_ids,
        )

        while not stop_event.is_set():
            loop_start = time.perf_counter()
            now = time.monotonic()

            # --- Grab all monitors ---
            frames: list[tuple[int, np.ndarray, np.ndarray, bool]] = []
            any_changed = False
            for mid in monitor_ids:
                monitor = monitor_meta[mid]["monitor"]
                frame_raw = np.array(sct.grab(monitor), dtype=np.uint8)
                frame_bgr = frame_raw[:, :, :3].copy()  # BGRA → BGR without cvtColor
                thumb = make_thumbnail(frame_bgr)
                changed = is_frame_changed(
                    prev_thumbs.get(mid), thumb, config.frame_change_threshold,
                )
                prev_thumbs[mid] = thumb
                any_changed = any_changed or changed
                frames.append((mid, frame_bgr, thumb, changed))

            # --- FIX #1: skip when no change regardless of overlay state ---
            if any_changed:
                active_until = max(active_until, now + config.active_seconds)
            else:
                target_fps = (
                    config.active_fps if now < active_until else config.idle_fps
                )
                _sleep_until_next(loop_start, target_fps, stop_event)
                continue

            # --- Detect + solve ---
            selected = _select_best_candidate(
                frames, config, reader, detect_cache, solve_cache, monitor_meta,
            )

            if selected is None:
                lost = tracker.on_no_grid()
                if (
                    overlay_visible
                    and lost >= config.lost_frames
                    and (now - last_valid_solution_at) >= config.overlay_hold_seconds
                ):
                    signals.clear_signal.emit()
                    LOGGER.debug(
                        "overlay cleared: no grid for %d frames", lost,
                    )
                    overlay_visible = False
                    last_render_key = None
                _sleep_until_next(
                    loop_start,
                    config.active_fps if now < active_until else config.idle_fps,
                    stop_event,
                )
                continue

            (sel_corners, sel_sig, sel_bbox, sel_payload,
             sel_mid, sel_offset, sel_scale) = selected

            stable = tracker.on_grid(sel_sig, sel_bbox, config)
            if stable < config.stable_frames:
                _sleep_until_next(
                    loop_start,
                    config.active_fps if now < active_until else config.idle_fps,
                    stop_event,
                )
                continue

            hint_cells = compute_hint_positions(
                sel_corners, sel_payload["hints"], sel_offset, sel_scale,
            )
            if not hint_cells:
                if overlay_visible:
                    signals.clear_signal.emit()
                    overlay_visible = False
                    last_render_key = None
                _sleep_until_next(
                    loop_start,
                    config.active_fps if now < active_until else config.idle_fps,
                    stop_event,
                )
                continue

            last_valid_solution_at = now
            render_scale = float((sel_scale[0] + sel_scale[1]) / 2.0)
            render_key = build_render_key(
                sel_sig, sel_bbox, render_scale, config.quantize_step,
            )
            if should_render(last_render_key, render_key):
                signals.update_signal.emit(hint_cells)
                LOGGER.debug(
                    "overlay updated: monitor=%s hints=%d stable=%d",
                    sel_mid, len(hint_cells), stable,
                )
                last_render_key = render_key
                overlay_visible = True

            _sleep_until_next(
                loop_start,
                config.active_fps if now < active_until else config.idle_fps,
                stop_event,
            )


def _select_best_candidate(
    frames: list[tuple[int, np.ndarray, np.ndarray, bool]],
    config: MonitorConfig,
    reader: CnnDigitReader,
    detect_cache: LruCache,
    solve_cache: LruCache,
    monitor_meta: dict[int, dict[str, Any]],
) -> Optional[tuple]:
    """Find the best solvable grid across all monitors/frames.

    Returns a tuple of (corners, signature, bbox, payload,
    monitor_id, logical_offset, capture_to_logical_scale) or None.
    """
    best_givens = -1
    best: Optional[tuple] = None

    for mid, frame_bgr, thumb, changed in frames:
        frame_key = f"m{mid}:{thumbnail_hash(thumb)}"
        candidates = detect_cache.get(frame_key)
        if candidates is None:
            candidates = detect_candidates_with_corners(frame_bgr)
            detect_cache.put(frame_key, candidates)

        if not candidates:
            continue

        for warped, corners in candidates:
            bbox = bbox_from_corners(corners)
            bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            frame_area = max(1, frame_bgr.shape[0] * frame_bgr.shape[1])
            if bbox_area / float(frame_area) < config.min_bbox_area_ratio:
                continue

            raw_sig = grid_signature(warped)
            cached_result = solve_cache.get(raw_sig)
            if cached_result is None:
                payload, reason, givens = solve_hints_from_warped(
                    warped, reader, config.min_givens,
                )
                solve_cache.put(raw_sig, {
                    "payload": payload, "reason": reason, "givens": int(givens),
                })
            else:
                payload = cached_result.get("payload")
                givens = int(cached_result.get("givens") or 0)

            if payload is None:
                continue

            givens = int(payload.get("givens") or givens or 0)
            if givens <= best_givens:
                continue

            meta = monitor_meta[mid]
            puz_sig = puzzle_signature(payload["original_grid"])
            best = (
                corners,
                f"m{mid}:g:{puz_sig}",
                bbox,
                payload,
                mid,
                tuple(meta["logical_offset"]),
                tuple(meta["capture_to_logical_scale"]),
            )
            best_givens = givens

    return best


def _sleep_until_next(
    loop_start: float,
    target_fps: float,
    stop_event: threading.Event,
) -> None:
    interval = 1.0 / max(float(target_fps), 0.1)
    elapsed = time.perf_counter() - loop_start
    remaining = max(0.0, interval - elapsed)
    stop_event.wait(remaining)
