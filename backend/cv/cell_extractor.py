"""Cell extraction from Sudoku grid."""

import cv2
import numpy as np
from typing import List


# Reusable CLAHE instance (avoid 81 allocations per grid).
_CLAHE = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))


def extract_cells(grid_image: np.ndarray, margin: int = 1) -> List[np.ndarray]:
    """
    Split a Sudoku grid image into 81 individual cell images.

    Args:
        grid_image: 450x450px grid image
        margin: Pixels trimmed from each cell side (skips grid lines)

    Returns:
        List of 81 cell images (row-major order)
    """
    # Ensure grid is 450x450
    if grid_image.shape[:2] != (450, 450):
        grid_image = cv2.resize(grid_image, (450, 450))

    cells = []
    cell_size = 50  # 450 / 9 = 50

    for row in range(9):
        for col in range(9):
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
