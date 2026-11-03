"""Cell extraction from Sudoku grid."""

import cv2
import numpy as np
from typing import List


# Reusable CLAHE instance (avoid 81 allocations per grid).
_CLAHE = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))

CELL_PX = 50  # warped grids are CELL_PX per cell: 450x450 for 9x9, 800x800 for 16x16


def extract_cells(grid_image: np.ndarray, margin: int = 1, size: int = 9) -> List[np.ndarray]:
    """
    Split a warped size x size grid into its cell images.

    Args:
        grid_image: CELL_PX * size pixels square
        margin: Pixels trimmed from each cell side (skips grid lines)
        size: Cells per side (9 or 16)

    Returns:
        size * size cell images (row-major order)
    """
    side = CELL_PX * size
    if grid_image.shape[:2] != (side, side):
        grid_image = cv2.resize(grid_image, (side, side))

    cells = []
    cell_size = CELL_PX

    for row in range(size):
        for col in range(size):
            x = col * cell_size
            y = row * cell_size

            cell = grid_image[y + margin : y + cell_size - margin, x + margin : x + cell_size - margin]

            cells.append(cell)

    return cells


def clean_cell(cell_image: np.ndarray, cell_size: int = 48) -> np.ndarray:
    """
    Clean and normalize a cell image for OCR.
    Simplified approach that preserves digit information.

    Args:
        cell_image: Raw cell image
        cell_size: Target output size

    Returns:
        Cleaned cell image (white background, black text)
    """
    # Convert to grayscale if needed
    if len(cell_image.shape) == 3:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_image.copy()

    # Resize to standard size first
    if gray.shape != (cell_size, cell_size):
        gray = cv2.resize(gray, (cell_size, cell_size))

    # Apply mild blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Enhance contrast with CLAHE (reuse module-level instance).
    enhanced = _CLAHE.apply(blurred)

    # Use Otsu thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Dilate slightly for thin digits
    cleaned = cv2.dilate(cleaned, kernel, iterations=1)

    # Invert back (white background, black text)
    result = cv2.bitwise_not(cleaned)

    return result
