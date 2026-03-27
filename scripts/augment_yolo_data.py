"""Augment YOLO grid detection dataset by pasting grids onto random backgrounds.

Takes existing labeled images, extracts the grid region, and composites it
onto generated backgrounds at random positions/scales/perspectives. This
multiplies the effective training set size.

Usage:
    uv run python scripts/augment_yolo_data.py [--count 500]
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
    """Generate a random background with varied texture."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)

    style = random.random()
    if style < 0.25:
        # Solid with gradient
        base = random.randint(20, 240)
        bg[:] = base
        grad = np.linspace(0, random.randint(-40, 40), h).reshape(-1, 1, 1)
        bg = np.clip(bg.astype(np.float32) + grad, 0, 255).astype(np.uint8)
    elif style < 0.5:
        # Multi-color patches (simulating a room/desk)
        for _ in range(random.randint(3, 12)):
            x1 = random.randint(0, w - 50)
            y1 = random.randint(0, h - 50)
            x2 = x1 + random.randint(50, min(500, w - x1))
            y2 = y1 + random.randint(30, min(400, h - y1))
            color = tuple(random.randint(0, 255) for _ in range(3))
            cv2.rectangle(bg, (x1, y1), (x2, y2), color, -1)
    elif style < 0.75:
        # Random circles and lines (simulating complex scene)
        base = random.randint(60, 200)
        bg[:] = base
        for _ in range(random.randint(5, 15)):
            cx = random.randint(0, w)
            cy = random.randint(0, h)
            r = random.randint(20, 200)
            color = tuple(random.randint(0, 255) for _ in range(3))
            cv2.circle(bg, (cx, cy), r, color, -1)
    else:
        # Perlin-like noise pattern
        base = np.random.randint(40, 220, (h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
        bg = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)

    # Add Gaussian noise
    noise = np.random.normal(0, random.uniform(5, 20), bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Occasional blur
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)

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
    """Generate random quadrilateral corners with perspective distortion.

    Returns 4 corners (TL, TR, BR, BL).
    """
    size = int(min(w, h) * grid_frac)
    margin_x = max(10, w - size - 10)
    margin_y = max(10, h - size - 10)

    x0 = random.randint(10, margin_x)
    y0 = random.randint(10, margin_y)

    # Perspective jitter: up to 10% of size (camera-like distortion)
    jitter = int(size * random.uniform(0.02, 0.10))

    def j():
        return random.randint(-jitter, jitter)

    corners = np.float32([
        [x0 + j(), y0 + j()],
        [x0 + size + j(), y0 + j()],
        [x0 + size + j(), y0 + size + j()],
        [x0 + j(), y0 + size + j()],
    ])

    corners[:, 0] = np.clip(corners[:, 0], 0, w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, h - 1)
    return corners


def augment_grid(grid: np.ndarray) -> np.ndarray:
    """Apply random augmentation to the grid image."""
    result = grid.copy()

    # Brightness/contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    # Gaussian blur (simulates camera out-of-focus)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    # Add noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(5, 15), result.shape)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return result


def corners_4_to_keypoints_16(corners: np.ndarray) -> np.ndarray:
    """Interpolate 4 corners [TL, TR, BR, BL] into 16 block-intersection keypoints."""
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
    keypoints = np.zeros((16, 2), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            u = col / 3.0
            v = row / 3.0
            top = tl * (1 - u) + tr * u
            bot = bl * (1 - u) + br * u
            keypoints[row * 4 + col] = top * (1 - v) + bot * v
    return keypoints


def corners_to_yolo_pose(keypoints: np.ndarray, img_w: int, img_h: int) -> str:
    """Convert keypoints (4 or 16) to YOLO pose format label."""
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for i in range(len(keypoints)):
        kx = float(keypoints[i, 0]) / img_w
        ky = float(keypoints[i, 1]) / img_h
        parts.append(f"{kx:.6f} {ky:.6f}")

    return " ".join(parts)


def load_source_grids(label_dir: Path, image_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load grid regions and their keypoints from labeled images.

    Returns list of (grid_450x450, normalized_keypoints) tuples.
    """
    results = []
    for lbl_path in sorted(label_dir.glob("*.txt")):
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = image_dir / f"{lbl_path.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        line = lbl_path.read_text().strip()
        parts = list(map(float, line.split()))
        h, w = image.shape[:2]

        n_values = len(parts)
        if n_values >= 37:  # 16 keypoints: 5 + 16*2
            n_kpt = 16
        elif n_values >= 13:  # 4 keypoints: 5 + 4*2
            n_kpt = 4
        else:
            continue

        keypoints = np.zeros((n_kpt, 2), dtype=np.float32)
        for i in range(n_kpt):
            keypoints[i, 0] = parts[5 + i * 2] * w
            keypoints[i, 1] = parts[5 + i * 2 + 1] * h

        if n_kpt == 16:
            corners_4 = keypoints[[0, 3, 15, 12]]  # TL, TR, BR, BL
        else:
            corners_4 = keypoints

        grid = extract_grid_region(image, corners_4)
        results.append((grid, keypoints / np.array([w, h], dtype=np.float32)))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Augment YOLO grid dataset")
    parser.add_argument("--count", type=int, default=500, help="Total images to generate")
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

    source_data = load_source_grids(lbl_dir, img_dir)
    if not source_data:
        print("No source grids found. Run auto_label_yolo.py first.")
        return 1
    print(f"Loaded {len(source_data)} source grids")

    w = h = args.size
    generated = 0

    for i in range(args.count):
        grid, _ = random.choice(source_data)
        grid = augment_grid(grid)
        bg = random_background(w, h)
        # Varied grid sizes: 20%-85% of image (camera photos have small grids)
        grid_frac = random.uniform(0.20, 0.85)
        corners = random_quad(w, h, grid_frac)
        result = paste_grid(bg, grid, corners)

        name = f"aug_{i:04d}"
        cv2.imwrite(str(img_dir / f"{name}.png"), result)
        kpts_16 = corners_4_to_keypoints_16(corners)
        label = corners_to_yolo_pose(kpts_16, w, h)
        (lbl_dir / f"{name}.txt").write_text(label + "\n")
        generated += 1

    print(f"Generated {generated} augmented images")
    print(f"Total dataset: {len(list(lbl_dir.glob('*.txt')))} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
