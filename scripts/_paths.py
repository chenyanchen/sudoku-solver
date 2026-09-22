"""Shared path utilities for training and evaluation scripts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_IMAGE_DIR = REPO_ROOT / "data" / "raw" / "images"
DEFAULT_CORNERS = REPO_ROOT / "data" / "labels" / "sudoku_corners.csv"


def load_corner_labels(path: Path) -> dict[str, np.ndarray]:
    """Read normalized TL/TR/BR/BL labels without relying on ignored image copies."""
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["image"]
            corners = np.array([
                [float(row[f"x_{corner}"]), float(row[f"y_{corner}"])]
                for corner in ("tl", "tr", "br", "bl")
            ], dtype=np.float32)
            if name in result or not np.isfinite(corners).all() or np.any((corners < 0) | (corners > 1)):
                raise ValueError(f"Duplicate or invalid corner label: {name}")
            result[name] = corners
    return result


def resolve_image_path(image_ref: str, labels_path: Path) -> Path:
    """Locate an image file from a label-relative reference string.

    Searches in order:
      1. Absolute path (if *image_ref* is absolute)
      2. Relative to the labels file directory
      3. Relative to the repository root
      4. By filename inside ``data/raw/images/``

    Raises ``FileNotFoundError`` when no candidate matches.
    """
    ref = Path(image_ref)
    candidates: list[Path] = []

    if ref.is_absolute():
        candidates.append(ref)
    else:
        candidates.append(labels_path.parent / ref)
        candidates.append(REPO_ROOT / ref)
        candidates.append(DATA_IMAGE_DIR / ref.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Image not found. ref={image_ref} searched=[{searched}]")
