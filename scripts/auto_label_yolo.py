"""Auto-label images for YOLO26-pose grid detection using existing OpenCV detector.

Generates YOLO pose labels (bbox + 4 corner keypoints) from images where
the existing `find_grid_with_corners` can detect the grid. Output is written
to `data/yolo_grid/` in YOLO pose format.

Usage:
    uv run python scripts/auto_label_yolo.py [--src data/raw/images]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cv.grid_detector import find_grid_with_corners


def corners_to_yolo_pose(corners: np.ndarray, img_w: int, img_h: int) -> str:
    """Convert 4 corners (TL, TR, BR, BL) to YOLO pose label line.

    Format: <cls> <cx> <cy> <bw> <bh> <kx1> <ky1> <kx2> <ky2> <kx3> <ky3> <kx4> <ky4>
    All values normalized to [0, 1].
    """
    xs = corners[:, 0]
    ys = corners[:, 1]

    # Bounding box from corners
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    # Keypoints: TL, TR, BR, BL (same order as find_grid_with_corners)
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(4):
        kx = float(corners[i, 0]) / img_w
        ky = float(corners[i, 1]) / img_h
        parts.append(f"{kx:.6f} {ky:.6f}")

    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label for YOLO grid detection")
    parser.add_argument(
        "--src",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "images",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=REPO_ROOT / "data" / "yolo_grid",
    )
    args = parser.parse_args()

    img_dir = args.dst / "images" / "train"
    lbl_dir = args.dst / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for img_path in sorted(args.src.glob("*.png")):
        image = cv2.imread(str(img_path))
        if image is None:
            failed.append((img_path.name, "unreadable"))
            continue

        result = find_grid_with_corners(image)
        if result is None:
            failed.append((img_path.name, "no grid detected"))
            continue

        _, corners = result
        h, w = image.shape[:2]
        label = corners_to_yolo_pose(corners, w, h)

        stem = img_path.stem
        shutil.copy2(img_path, img_dir / img_path.name)
        (lbl_dir / f"{stem}.txt").write_text(label + "\n")

        print(f"  OK  {img_path.name:35s} {w}x{h}  corners: {corners.tolist()}")
        success += 1

    print(f"\nLabeled: {success}, Failed: {len(failed)}")
    for name, reason in failed:
        print(f"  SKIP {name}: {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
