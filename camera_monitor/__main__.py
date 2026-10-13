"""CLI entry point: python -m camera_monitor."""

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

from backend.ocr.cnn_digit_reader import DEFAULT_CNN_MODEL

from .app import CameraConfig, run_camera


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime Sudoku camera solver (macOS Continuity Camera)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="cv2.VideoCapture index (iPhone Continuity Camera is usually 1+)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CNN_MODEL_PATH", str(DEFAULT_CNN_MODEL)),
        help="CNN OCR model path (fallback to CNN_MODEL_PATH env or default)",
    )
    parser.add_argument(
        "--yolo-model",
        default=os.getenv("YOLO_MODEL_PATH"),
        help="YOLO grid detector model path (fallback to default)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Target display FPS")
    parser.add_argument("--min-givens", type=int, default=17)
    parser.add_argument("--min-confidence", type=float, default=0.30,
                        help="Minimum OCR average confidence to accept (0.0-1.0)")
    parser.add_argument("--yolo-conf", type=float, default=0.3,
                        help="YOLO detection confidence threshold")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    config = CameraConfig(
        camera_index=args.camera,
        model_path=args.model,
        yolo_model_path=args.yolo_model,
        fps=args.fps,
        min_givens=args.min_givens,
        min_confidence=args.min_confidence,
        yolo_conf=args.yolo_conf,
        debug=args.debug,
    )
    return run_camera(config)


if __name__ == "__main__":
    raise SystemExit(main())
