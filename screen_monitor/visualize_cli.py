"""CV pipeline step-by-step visualiser CLI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2

LOGGER = logging.getLogger("screen_monitor.visualize_cli")


def run_visualize(
    image_path: str,
    output_dir: str,
    model_path: Optional[str] = None,
) -> int:
    """Run the grid detection pipeline on *image_path* and save step images.

    Returns 0 on success, 1 on failure.
    """
    from backend.cv.grid_detector import find_grid_with_debug

    img = cv2.imread(image_path)
    if img is None:
        LOGGER.error("Cannot read image: %s", image_path)
        return 1

    reader = None
    if model_path:
        try:
            from backend.ocr.cnn_digit_reader import CnnDigitReader

            reader = CnnDigitReader(model_path=model_path, strict=False)
            if not reader.is_ready:
                LOGGER.warning("CNN model failed to load, skipping OCR annotation")
                reader = None
        except Exception:
            LOGGER.warning("Could not load CNN reader, skipping OCR annotation")

    debug = find_grid_with_debug(img, reader=reader)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    steps = [
        ("01_original", debug.original),
        ("02_gray", debug.gray),
        ("03_thresh", debug.thresh),
        ("04_morphed", debug.morphed),
        ("05_contours", debug.contours_image),
        ("06_quad", debug.quad_image),
        ("07_warped", debug.warped),
        ("08_cells", debug.cells_montage),
    ]

    saved = 0
    for name, mat in steps:
        if mat is not None:
            path = out / f"{name}.png"
            cv2.imwrite(str(path), mat)
            LOGGER.info("Saved %s", path)
            saved += 1

    if debug.error:
        LOGGER.warning("Pipeline stopped early: %s", debug.error)

    LOGGER.info("Saved %d / %d step images to %s", saved, len(steps), out)
    return 0
