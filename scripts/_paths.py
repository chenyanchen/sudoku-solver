"""Shared path utilities for training and evaluation scripts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_IMAGE_DIR = REPO_ROOT / "data" / "raw" / "images"


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
