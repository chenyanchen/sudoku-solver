"""Image preprocessing for Sudoku grid detection."""

import cv2
import numpy as np


def preprocess(image: np.ndarray) -> dict:
    """
    Preprocess image for Sudoku grid detection.

    Args:
        image: Input image (BGR format from OpenCV)

    Returns:
        Dictionary containing:
            - gray: Grayscale image
            - thresh: Thresholded binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    # This handles varying lighting conditions better than simple thresholding
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # Block size (must be odd)
        2,  # Constant subtracted from mean
    )

    return {"gray": gray, "thresh": thresh}


def morphological_operations(thresh: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply morphological operations to clean up thresholded image.

    Args:
        thresh: Binary thresholded image
        kernel_size: Size of morphological kernel

    Returns:
        Processed binary image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Close small holes
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Remove small noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Dilate to make grid lines thicker
    dilated = cv2.dilate(opened, kernel, iterations=1)

    return dilated
