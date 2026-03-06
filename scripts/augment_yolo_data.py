"""Augment YOLO grid detection dataset by pasting grids onto random backgrounds.

Takes existing labeled images, extracts the grid region, and composites it
onto generated backgrounds at random positions/scales/perspectives. This
multiplies the effective training set size.

Usage:
    uv run python scripts/augment_yolo_data.py [--count 200]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def random_background(w: int, h: int) -> np.ndarray:
    """Generate a random background that looks like a screen/desktop."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    # Random base color (light or dark theme)
    if random.random() < 0.5:
        base = random.randint(200, 255)
    else:
        base = random.randint(20, 80)
    bg[:] = base

    # Add some random rectangles (simulating UI elements)
    for _ in range(random.randint(2, 8)):
        x1 = random.randint(0, w - 50)
        y1 = random.randint(0, h - 50)
        x2 = x1 + random.randint(50, min(400, w - x1))
        y2 = y1 + random.randint(30, min(300, h - y1))
        color = tuple(random.randint(0, 255) for _ in range(3))
        cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)

    # Add noise
    noise = np.random.normal(0, 8, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return bg


def extract_grid_region(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Extract the grid as a 450x450 warped image."""
    dst = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(image, matrix, (450, 450))


def paste_grid(
    bg: np.ndarray, grid_img: np.ndarray, target_corners: np.ndarray
) -> np.ndarray:
    """Paste grid_img onto bg using perspective transform to target_corners."""
    src = np.float32([[0, 0], [449, 0], [449, 449], [0, 449]])
    matrix = cv2.getPerspectiveTransform(src, target_corners.astype(np.float32))
    warped = cv2.warpPerspective(grid_img, matrix, (bg.shape[1], bg.shape[0]))
    mask = cv2.warpPerspective(
        np.ones_like(grid_img) * 255, matrix, (bg.shape[1], bg.shape[0])
    )
    mask_f = mask.astype(np.float32) / 255.0
    result = (bg.astype(np.float32) * (1 - mask_f) + warped.astype(np.float32) * mask_f)
    return result.astype(np.uint8)


def random_quad(w: int, h: int, grid_frac: float) -> np.ndarray:
    """Generate random quadrilateral corners for placing a grid.

    Returns 4 corners (TL, TR, BR, BL) with slight perspective distortion.
    """
    size = int(min(w, h) * grid_frac)
    margin_x = max(10, w - size - 10)
    margin_y = max(10, h - size - 10)

    x0 = random.randint(10, margin_x)
    y0 = random.randint(10, margin_y)

    # Add slight perspective jitter (up to 3% of size)
    jitter = int(size * 0.03)

    def j():
        return random.randint(-jitter, jitter)

    corners = np.float32([
        [x0 + j(), y0 + j()],           # TL
        [x0 + size + j(), y0 + j()],     # TR
        [x0 + size + j(), y0 + size + j()],  # BR
        [x0 + j(), y0 + size + j()],     # BL
    ])

    # Clip to image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)

    return corners


def corners_to_yolo_pose(corners: np.ndarray, img_w: int, img_h: int) -> str:
    """Convert corners to YOLO pose format label."""
    xs = corners[:, 0]
    ys = corners[:, 1]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(4):
        kx = float(corners[i, 0]) / img_w
        ky = float(corners[i, 1]) / img_h
        parts.append(f"{kx:.6f} {ky:.6f}")

    return " ".join(parts)


def load_source_grids(label_dir: Path, image_dir: Path) -> list[np.ndarray]:
    """Load grid regions from already-labeled images."""
    grids = []
    for lbl_path in sorted(label_dir.glob("*.txt")):
        img_path = image_dir / f"{lbl_path.stem}.png"
        if not img_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        line = lbl_path.read_text().strip()
        parts = list(map(float, line.split()))
        # parts: cls cx cy bw bh kx1 ky1 kx2 ky2 kx3 ky3 kx4 ky4
        h, w = image.shape[:2]
        corners = np.float32([
            [parts[5] * w, parts[6] * h],
            [parts[7] * w, parts[8] * h],
            [parts[9] * w, parts[10] * h],
            [parts[11] * w, parts[12] * h],
        ])
        grid = extract_grid_region(image, corners)
        grids.append(grid)

    return grids


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment YOLO grid dataset")
    parser.add_argument("--count", type=int, default=200, help="Total images to generate")
    parser.add_argument("--size", type=int, default=640, help="Output image size")
    parser.add_argument(
        "--dst",
        type=Path,
        default=REPO_ROOT / "data" / "yolo_grid",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    img_dir = args.dst / "images" / "train"
    lbl_dir = args.dst / "labels" / "train"

    # Load source grids from existing labels
    grids = load_source_grids(lbl_dir, img_dir)
    if not grids:
        print("No source grids found. Run auto_label_yolo.py first.")
        return 1

    print(f"Loaded {len(grids)} source grids")

    w = h = args.size
    generated = 0

    for i in range(args.count):
        grid = random.choice(grids)

        # Random augmentation on the grid itself
        if random.random() < 0.3:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            grid = cv2.convertScaleAbs(grid, alpha=alpha, beta=beta)

        bg = random_background(w, h)
        grid_frac = random.uniform(0.3, 0.85)
        corners = random_quad(w, h, grid_frac)
        result = paste_grid(bg, grid, corners)

        name = f"aug_{i:04d}"
        cv2.imwrite(str(img_dir / f"{name}.png"), result)
        label = corners_to_yolo_pose(corners, w, h)
        (lbl_dir / f"{name}.txt").write_text(label + "\n")
        generated += 1

    print(f"Generated {generated} augmented images")
    print(f"Total dataset: {len(list(lbl_dir.glob('*.txt')))} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
