"""CLI entry point: python -m screen_monitor."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so backend imports work.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from .agent import run_monitor
from .types import MonitorConfig


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--rescan-interval", type=float, default=1.0,
                        help="Seconds between forced rescans when frame is unchanged")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--debug-hud", action="store_true",
        help="Show real-time debug HUD overlay (FPS, latency, cache stats)",
    )
    parser.add_argument(
        "--visualize",
        metavar="IMAGE",
        help="Run CV pipeline visualisation on IMAGE and exit (mutually exclusive with monitor mode)",
    )
    parser.add_argument(
        "--viz-output",
        metavar="DIR",
        default="viz_output",
        help="Output directory for --visualize step images (default: viz_output)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # --visualize mode: run pipeline visualiser and exit.
    if args.visualize:
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            force=True,
        )
        from .visualize_cli import run_visualize

        return run_visualize(args.visualize, args.viz_output, model_path=args.model)

    config = MonitorConfig(
        model_path=args.model,
        monitor_index=args.monitor_index,
        idle_fps=args.idle_fps,
        active_fps=args.active_fps,
        active_seconds=args.active_seconds,
        stable_frames=args.stable_frames,
        lost_frames=args.lost_frames,
        min_bbox_area_ratio=args.min_bbox_area_ratio,
        overlay_hold_seconds=args.overlay_hold_seconds,
        rescan_interval=args.rescan_interval,
        debug=args.debug,
        debug_hud=args.debug_hud,
    )
    return run_monitor(config)


if __name__ == "__main__":
    raise SystemExit(main())
