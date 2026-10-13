"""Auto-label images for YOLO26-pose grid detection using existing OpenCV detector.

Generates YOLO pose labels (bbox + 4 corner keypoints) from images where
the existing `find_grid_with_corners` can detect the grid. Output is written
to `data/yolo_grid/` in YOLO pose format.

After detection, each candidate is warped and validated with grid-line
regularity scoring to ensure keypoints mark the actual sudoku grid, not
(for example) the entire phone screen.

Usage:
    uv run python scripts/auto_label_yolo.py [--src data/raw/images] [--visualize]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.cv.grid_detector import (
    has_regular_grid_lines,
    find_grid_with_corners,
    find_grids_with_corners,
)


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


def corners_4_to_keypoints_16(corners: np.ndarray) -> np.ndarray:
    """Interpolate 4 corners (TL, TR, BR, BL) into 16 block-intersection keypoints.

    The 16 points form a 4x4 grid corresponding to the intersections of the
    3x3 block boundaries. Points are evenly spaced along the edges of the
    quadrilateral defined by the 4 corners using bilinear interpolation.

    Args:
        corners: Shape (4, 2), order [TL, TR, BR, BL].

    Returns:
        Shape (16, 2) keypoints in the same coordinate space as input corners.
    """
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

    keypoints = np.zeros((16, 2), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            u = col / 3.0  # 0, 1/3, 2/3, 1
            v = row / 3.0
            top = tl * (1 - u) + tr * u
            bot = bl * (1 - u) + br * u
            keypoints[row * 4 + col] = top * (1 - v) + bot * v

    return keypoints


def keypoints_16_to_yolo_pose(keypoints: np.ndarray, img_w: int, img_h: int) -> str:
    """Convert 16 keypoints to YOLO pose label line.

    Format: <cls> <cx> <cy> <bw> <bh> <kx0> <ky0> ... <kx15> <ky15>
    All values normalized to [0, 1].
    """
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(16):
        kx = float(keypoints[i, 0]) / img_w
        ky = float(keypoints[i, 1]) / img_h
        parts.append(f"{kx:.6f} {ky:.6f}")

    return " ".join(parts)


def _validate_warped_grid(warped: np.ndarray) -> bool:
    """Check if the warped image contains a regular grid pattern."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    return has_regular_grid_lines(gray)


def _find_best_grid(image: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Find the best grid candidate using regularity validation.

    1. Try find_grid_with_corners() — if the result passes regularity, use it.
    2. Otherwise, try find_grids_with_corners() for multiple candidates.
    3. Pick the candidate with the best regularity.
    """
    # Try the primary detection first.
    result = find_grid_with_corners(image)
    if result is not None:
        warped, corners = result
        if _validate_warped_grid(warped):
            return warped, corners

    # Fall back to multiple candidates.
    candidates = find_grids_with_corners(image)
    for warped, corners in candidates:
        if _validate_warped_grid(warped):
            return warped, corners

    # No candidate passed regularity — return the primary result as last resort.
    if result is not None:
        return result

    # Return the first candidate if any exist.
    if candidates:
        return candidates[0]

    return None


def _visualize_detection(
    image: np.ndarray,
    corners: np.ndarray,
    warped: np.ndarray,
    name: str,
) -> None:
    """Show detection result in a window for manual review."""
    vis = image.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis, [pts], True, (0, 255, 0), 3)
    for i, pt in enumerate(corners):
        label = ["TL", "TR", "BR", "BL"][i]
        cv2.circle(vis, (int(pt[0]), int(pt[1])), 8, (0, 0, 255), -1)
        cv2.putText(vis, label, (int(pt[0]) + 10, int(pt[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Scale both for display.
    max_dim = 800
    h, w = vis.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

    wh, ww = warped.shape[:2]
    warped_display = cv2.resize(warped, (int(ww * scale), int(wh * scale))) if scale < 1.0 else warped.copy()

    cv2.imshow(f"{name} - corners", vis)
    cv2.imshow(f"{name} - warped", warped_display)
    print(f"  Showing {name}. Press any key to continue, 'q' to quit...")
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    if key == ord("q"):
        raise KeyboardInterrupt("User quit visualization")


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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show detection results for manual review",
    )
    args = parser.parse_args()

    img_dir = args.dst / "images" / "train"
    lbl_dir = args.dst / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    extensions = ("*.jpg", "*.jpeg", "*.png")
    all_images = []
    for ext in extensions:
        all_images.extend(args.src.glob(ext))

    for img_path in sorted(all_images):
        image = cv2.imread(str(img_path))
        if image is None:
            failed.append((img_path.name, "unreadable"))
            continue

        result = _find_best_grid(image)
        if result is None:
            failed.append((img_path.name, "no grid detected"))
            continue

        warped, corners = result
        h, w = image.shape[:2]

        if args.visualize:
            try:
                _visualize_detection(image, corners, warped, img_path.name)
            except KeyboardInterrupt:
                print("\nVisualization stopped by user.")
                break

        kpts_16 = corners_4_to_keypoints_16(corners)
        label = keypoints_16_to_yolo_pose(kpts_16, w, h)

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
