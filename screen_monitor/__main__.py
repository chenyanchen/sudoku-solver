"""CLI entry point: python -m screen_monitor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so backend imports work.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .agent import run_monitor
from .types import MonitorConfig


def _parse_args() -> MonitorConfig:
    parser = argparse.ArgumentParser(
        description="Realtime Sudoku screen monitor (macOS)",
    )
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
        help="MSS monitor index: 0=all, 1/2/...=specific monitor",
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


def main() -> int:
    config = _parse_args()
    return run_monitor(config)


if __name__ == "__main__":
    raise SystemExit(main())
