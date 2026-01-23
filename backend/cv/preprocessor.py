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
            - blurred: Gaussian blurred image
            - thresh: Thresholded binary image
            - original: Original image (copy)
    """
    if image is None or len(image) == 0:
        raise ValueError("Invalid input image")

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

    return {
        "gray": gray,
        "blurred": blurred,
        "thresh": thresh,
        "original": image.copy(),
    }


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.

    Args:
        image: Grayscale input image

    Returns:
        Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def denoise(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from image using non-local means denoising.

    Args:
        image: Input image (grayscale)

    Returns:
        Denoised image
    """
    return cv2.fastNlMeansDenoising(
        image, None, h=10, templateWindowSize=7, searchWindowSize=21
    )


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


def invert_image(image: np.ndarray) -> np.ndarray:
    """
    Invert a grayscale image.

    Args:
        image: Input image

    Returns:
        Inverted image
    """
    return cv2.bitwise_not(image)
