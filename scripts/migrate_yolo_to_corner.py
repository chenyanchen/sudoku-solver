"""One-time migration: convert 16-keypoint YOLO labels to 4-corner CSV.

Usage:
    uv run python scripts/migrate_yolo_to_corner.py
"""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_LABELS = REPO_ROOT / "data" / "yolo_grid" / "labels" / "train"
YOLO_IMAGES = REPO_ROOT / "data" / "yolo_grid" / "images" / "train"
CORNER_DIR = REPO_ROOT / "data" / "corner_cnn"
CORNER_IMAGES = CORNER_DIR / "images"
CORNER_CSV = CORNER_DIR / "labels.csv"

# In 16-kpt YOLO format, keypoints are at indices 0-15 (row-major 4x4).
# Outer corners: P00=0 (TL), P03=3 (TR), P33=15 (BR), P30=12 (BL)
_TL, _TR, _BR, _BL = 0, 3, 15, 12


def main() -> int:
    CORNER_IMAGES.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for label_path in sorted(YOLO_LABELS.glob("*.txt")):
        stem = label_path.stem
        # Find matching image
        image_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = YOLO_IMAGES / f"{stem}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            print(f"  SKIP {stem}: no matching image")
            continue

        # Parse YOLO label: class cx cy w h kx0 ky0 kx1 ky1 ... kx15 ky15
        parts = label_path.read_text().strip().split()
        if len(parts) != 37:
            print(f"  SKIP {stem}: expected 37 parts, got {len(parts)}")
            continue

        # Extract 16 keypoints (normalized [0, 1])
        kpts = []
        for i in range(16):
            kx = float(parts[5 + i * 2])
            ky = float(parts[5 + i * 2 + 1])
            kpts.append((kx, ky))

        tl = kpts[_TL]
        tr = kpts[_TR]
        br = kpts[_BR]
        bl = kpts[_BL]

        # Symlink image
        dst = CORNER_IMAGES / image_path.name
        if not dst.exists():
            dst.symlink_to(image_path.resolve())

        rows.append({
            "image": image_path.name,
            "x_tl": f"{tl[0]:.6f}", "y_tl": f"{tl[1]:.6f}",
            "x_tr": f"{tr[0]:.6f}", "y_tr": f"{tr[1]:.6f}",
            "x_br": f"{br[0]:.6f}", "y_br": f"{br[1]:.6f}",
            "x_bl": f"{bl[0]:.6f}", "y_bl": f"{bl[1]:.6f}",
        })

    # Write CSV
    fieldnames = ["image", "x_tl", "y_tl", "x_tr", "y_tr", "x_br", "y_br", "x_bl", "y_bl"]
    with CORNER_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Migrated {len(rows)} samples to {CORNER_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
