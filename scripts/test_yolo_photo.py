"""Test YOLO grid detector + CNN OCR + solver on a single image.

Usage:
    uv run python scripts/test_yolo_photo.py data/raw/images/photo_1.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.ocr.cnn_digit_reader import CnnDigitReader
from camera_monitor.yolo_detector import YoloGridDetector
from backend.ocr.pipeline import recognize


def main() -> int:
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/images/photo_1.jpg"
    image = cv2.imread(str(REPO_ROOT / img_path))
    if image is None:
        print(f"Cannot read: {img_path}")
        return 1

    print(f"Image: {img_path} ({image.shape[1]}x{image.shape[0]})")

    # YOLO detection.
    detector = YoloGridDetector()
    if not detector.is_ready:
        print(f"YOLO model error: {detector.load_error}")
        return 1

    det = detector.detect(image)
    if det is None:
        print("YOLO: no grid detected")
        return 1

    warped, corners, conf = det
    print(f"YOLO: detected grid (conf={conf:.3f})")
    print(f"  corners: {corners.tolist()}")

    # Show warped grid.
    cv2.imwrite("/tmp/yolo_warped.png", warped)
    print("  warped saved to /tmp/yolo_warped.png")

    # OCR + solve.
    reader = CnnDigitReader()
    if not reader.is_ready:
        print(f"CNN model error: {reader.load_error}")
        return 1

    result = recognize(reader, warped)
    print(f"  OCR: {result.status}, givens={result.givens}")

    if result.solution is None:
        print("  Solve failed")
        return 1

    print(f"  Solved! {len(result.hints)} hints")
    print("\n  Original grid:")
    for row in result.grid:
        print("   ", " ".join(str(d) if d else "." for d in row))

    print("\n  Solution:")
    for row in result.solution:
        print("   ", " ".join(str(d) for d in row))

    # Draw result.
    vis = image.copy()
    pts = corners.astype(int).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    cv2.imwrite("/tmp/yolo_result.png", vis)
    print("\n  Result saved to /tmp/yolo_result.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
