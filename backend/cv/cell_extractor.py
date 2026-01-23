"""Cell extraction from Sudoku grid."""

import cv2
import numpy as np
from typing import List, Tuple


def extract_cells(grid_image: np.ndarray) -> List[np.ndarray]:
    """
    Split a Sudoku grid image into 81 individual cell images.

    Args:
        grid_image: 450x450px grid image

    Returns:
        List of 81 cell images (row-major order)
    """
    if grid_image is None:
        return []

    # Ensure grid is 450x450
    if grid_image.shape[:2] != (450, 450):
        grid_image = cv2.resize(grid_image, (450, 450))

    cells = []
    cell_size = 50  # 450 / 9 = 50

    for row in range(9):
        for col in range(9):
            # Extract cell with some margin for border removal
            x = col * cell_size
            y = row * cell_size

            # Extract cell (add 1px to skip grid line, subtract 2px for other border)
            cell = grid_image[y + 1:y + cell_size - 1, x + 1:x + cell_size - 1]

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
    if cell_image is None or cell_image.size == 0:
        # Return blank white image
        return np.ones((cell_size, cell_size), dtype=np.uint8) * 255

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

    # Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(blurred)

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


def extract_digit(cell_image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Extract and center the digit from a cell image.

    Args:
        cell_image: Cleaned cell image

    Returns:
        Tuple of (processed_image, has_digit)
    """
    if cell_image is None or cell_image.size == 0:
        return np.ones((48, 48), dtype=np.uint8) * 255, False

    # Convert to grayscale if needed
    if len(cell_image.shape) == 3:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_image.copy()

    # Threshold to separate digit from background
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours of the digit
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.ones((48, 48), dtype=np.uint8) * 255, False

    # Get the largest contour (the digit)
    digit_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(digit_contour)

    # Check if this is actually a digit (has enough area)
    if area < 30:  # Too small, likely noise
        return np.ones((48, 48), dtype=np.uint8) * 255, False

    # Get bounding box
    x, y, w, h = cv2.boundingRect(digit_contour)

    # Add some padding
    padding = 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(gray.shape[1] - x, w + 2 * padding)
    h = min(gray.shape[0] - y, h + 2 * padding)

    # Extract the digit
    digit = gray[y:y + h, x:x + w]

    # Create a centered image
    output_size = 48
    centered = np.ones((output_size, output_size), dtype=np.uint8) * 255

    # Calculate position to center the digit
    if w > 0 and h > 0:
        # Resize digit to fit within output while maintaining aspect ratio
        scale = min((output_size - 4) / w, (output_size - 4) / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w > 0 and new_h > 0:
            digit_resized = cv2.resize(digit, (new_w, new_h))

            # Center the digit
            start_x = (output_size - new_w) // 2
            start_y = (output_size - new_h) // 2

            centered[start_y:start_y + new_h, start_x:start_x + new_w] = digit_resized

    return centered, True


def get_cell_grid_positions(grid_size: int = 9) -> List[Tuple[int, int, int, int]]:
    """
    Get the grid positions for extracting cells.

    Args:
        grid_size: Size of the grid (default 9 for Sudoku)

    Returns:
        List of (x, y, width, height) for each cell
    """
    positions = []
    cell_width = 450 // grid_size
    cell_height = 450 // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_width
            y = row * cell_height
            positions.append((x, y, cell_width, cell_height))

    return positions


def extract_cells_with_positions(
    grid_image: np.ndarray,
    grid_size: int = 9
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Extract cells along with their grid positions.

    Args:
        grid_image: Grid image
        grid_size: Size of the grid

    Returns:
        List of (cell_image, row, col) tuples
    """
    if grid_image is None:
        return []

    cells_with_pos = []
    cell_size = 450 // grid_size

    for row in range(grid_size):
        for col in range(grid_size):
            x = col * cell_size
            y = row * cell_size

            # Extract cell with border removal
            cell = grid_image[y + 1:y + cell_size - 1, x + 1:x + cell_size - 1]
            cells_with_pos.append((cell, row, col))

    return cells_with_pos
