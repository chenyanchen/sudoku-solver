"""OCR digit recognition using Tesseract."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, List


# Tesseract configuration for single digit recognition
DIGIT_CONFIG = '--psm 10 --oem 3 -c tessedit_char_whitelist=123456789'
# psm 10: Treat image as a single character
# oem 3: Use both LSTM and legacy engines


class DigitReader:
    """Read digits from Sudoku cell images using Tesseract OCR."""

    def __init__(self, config: str = DIGIT_CONFIG):
        """
        Initialize the digit reader.

        Args:
            config: Tesseract configuration string
        """
        self.config = config

    def recognize_digit(self, cell_image: np.ndarray, threshold: float = 30.0) -> Optional[int]:
        """
        Recognize a single digit from a cell image.
        Uses multiple preprocessing attempts for faint/light digits.

        Args:
            cell_image: Grayscale cell image
            threshold: Confidence threshold for recognition (lowered for faint digits)

        Returns:
            Recognized digit (1-9) or None if empty/uncertain
        """
        if cell_image is None or cell_image.size == 0:
            return None

        # Try multiple preprocessing approaches
        for processed in self._prepare_image_variants(cell_image):
            try:
                # Get OCR data with confidence
                data = pytesseract.image_to_data(
                    processed,
                    config=self.config,
                    output_type=pytesseract.Output.DICT
                )

                # Extract text and confidence
                text = data.get('text', [])
                conf = data.get('conf', [])

                if not text or not conf:
                    continue

                # Iterate through all entries to find the first valid digit
                # Tesseract may return multiple entries; actual digit is often not at index 0
                for char, confidence in zip(text, conf):
                    char = char.strip()
                    # Check if we got a valid digit with sufficient confidence
                    if char.isdigit() and 1 <= int(char) <= 9:
                        if confidence >= threshold:
                            return int(char)

            except Exception:
                continue

        return None

    def recognize_grid(
        self,
        cells: List[np.ndarray],
        threshold: float = 50.0
    ) -> List[List[int]]:
        """
        Recognize digits from a list of 81 cell images.

        Args:
            cells: List of 81 cell images (row-major order)
            threshold: Confidence threshold for recognition

        Returns:
            9x9 grid with recognized digits (0 for empty cells)
        """
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells, got {len(cells)}")

        grid = []
        for row in range(9):
            grid_row = []
            for col in range(9):
                idx = row * 9 + col
                digit = self.recognize_digit(cells[idx], threshold)
                grid_row.append(digit if digit else 0)
            grid.append(grid_row)

        return grid

    def _prepare_image(self, cell_image: np.ndarray) -> np.ndarray:
        """
        Prepare cell image for OCR.

        Args:
            cell_image: Input cell image

        Returns:
            Processed image ready for Tesseract
        """
        # Ensure grayscale
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        # Resize to larger size for better OCR
        target_size = 64
        if gray.shape[0] != target_size or gray.shape[1] != target_size:
            gray = cv2.resize(gray, (target_size, target_size))

        # Apply threshold for better contrast
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _prepare_image_variants(self, cell_image: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple preprocessed variants of the cell image.
        Tries different approaches to handle various digit styles.

        Args:
            cell_image: Input cell image

        Returns:
            List of processed image variants
        """
        variants = []

        # Ensure grayscale
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        # Target size for OCR
        target_size = 64

        # Variant 1: Standard Otsu threshold
        resized1 = cv2.resize(gray, (target_size, target_size))
        _, binary1 = cv2.threshold(resized1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(binary1)

        # Variant 2: Aggressive adaptive threshold for faint digits
        resized2 = cv2.resize(gray, (target_size, target_size))
        blurred2 = cv2.GaussianBlur(resized2, (3, 3), 0)
        clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced2 = clahe2.apply(blurred2)
        adaptive2 = cv2.adaptiveThreshold(
            enhanced2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 8
        )
        variants.append(adaptive2)

        # Variant 3: Invert and try (for dark digits on light background)
        _, binary3 = cv2.threshold(resized1, 127, 255, cv2.THRESH_BINARY)
        variants.append(binary3)

        # Variant 4: Dilated to make thin digits thicker
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(adaptive2, kernel, iterations=1)
        variants.append(dilated)

        # Variant 5: High contrast stretch
        resized5 = cv2.resize(gray, (target_size, target_size))
        min_val = np.min(resized5)
        max_val = np.max(resized5)
        if max_val > min_val:
            stretched = ((resized5 - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            stretched = resized5
        _, binary5 = cv2.threshold(stretched, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        variants.append(binary5)

        # Ensure all have white background (invert if needed)
        final_variants = []
        for v in variants:
            mean_val = np.mean(v)
            if mean_val < 127:
                v = cv2.bitwise_not(v)
            final_variants.append(v)

        return final_variants

    def recognize_with_fallback(
        self,
        cell_image: np.ndarray,
        threshold: float = 50.0
    ) -> Optional[int]:
        """
        Recognize a digit with multiple preprocessing attempts.

        Args:
            cell_image: Input cell image
            threshold: Confidence threshold

        Returns:
            Recognized digit or None
        """
        # Try standard preprocessing
        result = self.recognize_digit(cell_image, threshold)
        if result is not None:
            return result

        # Try with inverted image
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image

        inverted = cv2.bitwise_not(gray)
        result = self.recognize_digit(inverted, threshold)
        if result is not None:
            return result

        # Try with blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        result = self.recognize_digit(blurred, threshold)
        if result is not None:
            return result

        return None


def recognize_digit(cell_image: np.ndarray, threshold: float = 50.0) -> Optional[int]:
    """Convenience function to recognize a single digit."""
    reader = DigitReader()
    return reader.recognize_digit(cell_image, threshold)


def recognize_grid(cells: List[np.ndarray], threshold: float = 50.0) -> List[List[int]]:
    """Convenience function to recognize a full grid."""
    reader = DigitReader()
    return reader.recognize_grid(cells, threshold)


def preprocess_for_ocr(cell_image: np.ndarray, target_size: int = 64) -> np.ndarray:
    """
    Preprocess a cell image specifically for Tesseract OCR.

    Args:
        cell_image: Input cell image
        target_size: Target size for output image

    Returns:
        Preprocessed image
    """
    if cell_image is None or cell_image.size == 0:
        return np.ones((target_size, target_size), dtype=np.uint8) * 255

    # Convert to grayscale
    if len(cell_image.shape) == 3:
        gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_image.copy()

    # Resize
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size))

    # Apply threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Ensure white background
    mean_val = np.mean(binary)
    if mean_val < 127:
        binary = cv2.bitwise_not(binary)

    return binary
