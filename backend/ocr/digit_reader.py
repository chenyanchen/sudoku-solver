"""OCR digit recognition using Tesseract."""

import cv2
import numpy as np
import pytesseract
from typing import Optional, List

from ..cv.cell_extractor import clean_cell, extract_digit


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
        Uses staged preprocessing attempts for faint/light digits.

        Args:
            cell_image: Grayscale cell image
            threshold: Confidence threshold for recognition (lowered for faint digits)

        Returns:
            Recognized digit (1-9) or None if empty/uncertain
        """
        if cell_image is None or cell_image.size == 0:
            return None

        # Fast blank check to avoid unnecessary OCR calls
        if self._is_likely_blank(cell_image):
            return None

        # Normalize and extract digit to reduce background noise
        normalized = clean_cell(cell_image, cell_size=48)
        digit_image, has_digit = extract_digit(normalized)
        candidate = digit_image if has_digit else normalized

        # Try fast preprocessing variants first
        for processed in self._prepare_fast_variants(candidate):
            result = self._run_ocr(processed, threshold)
            if result is not None:
                return result

        # Fall back to more aggressive variants if needed (only if a digit was found)
        if not has_digit:
            return None

        for processed in self._prepare_fallback_variants(digit_image):
            result = self._run_ocr(processed, threshold)
            if result is not None:
                return result

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

    def _is_likely_blank(self, cell_image: np.ndarray) -> bool:
        """
        Quick heuristic to skip OCR for empty cells.

        Uses low-variance detection on a downscaled grayscale image to avoid
        spending time on clearly blank cells.
        """
        if cell_image is None or cell_image.size == 0:
            return True

        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image

        # Downscale for speed
        small = cv2.resize(gray, (24, 24)) if gray.shape[:2] != (24, 24) else gray
        if float(np.std(small)) < 4.0:
            return True

        # Secondary check: very low ink ratio after Otsu thresholding
        _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_ratio = float(np.mean(binary > 0))
        return ink_ratio < 0.015

    def _run_ocr(self, processed: np.ndarray, threshold: float) -> Optional[int]:
        """
        Run Tesseract OCR on a processed image and return a digit if confident.
        """
        try:
            data = pytesseract.image_to_data(
                processed,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )

            text = data.get('text', [])
            conf = data.get('conf', [])

            if not text or not conf:
                return None

            for char, confidence in zip(text, conf):
                char = char.strip()
                if char.isdigit() and 1 <= int(char) <= 9:
                    try:
                        conf_value = float(confidence)
                    except (TypeError, ValueError):
                        continue
                    if conf_value >= threshold:
                        return int(char)
        except Exception:
            return None

        return None

    def _prepare_fast_variants(self, cell_image: np.ndarray) -> List[np.ndarray]:
        """
        Generate fast preprocessing variants.

        These are cheap and cover the common cases.
        """
        variants: List[np.ndarray] = []

        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        target_size = 64
        if gray.shape[:2] != (target_size, target_size):
            gray = cv2.resize(gray, (target_size, target_size))

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        variants.append(binary)

        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        variants.append(dilated)

        return variants

    def _prepare_fallback_variants(self, cell_image: np.ndarray) -> List[np.ndarray]:
        """
        Generate more aggressive variants as a fallback.
        """
        variants: List[np.ndarray] = []

        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image.copy()

        target_size = 64
        if gray.shape[:2] != (target_size, target_size):
            gray = cv2.resize(gray, (target_size, target_size))

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 5
        )
        if np.mean(adaptive) < 127:
            adaptive = cv2.bitwise_not(adaptive)
        variants.append(adaptive)

        # High-contrast stretch
        min_val = np.min(gray)
        max_val = np.max(gray)
        if max_val > min_val:
            stretched = ((gray - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            stretched = gray
        _, stretched_bin = cv2.threshold(
            stretched, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if np.mean(stretched_bin) < 127:
            stretched_bin = cv2.bitwise_not(stretched_bin)
        variants.append(stretched_bin)

        return variants

    def _prepare_image_variants(self, cell_image: np.ndarray) -> List[np.ndarray]:
        """
        Generate multiple preprocessed variants of the cell image.
        Tries different approaches to handle various digit styles.

        Args:
            cell_image: Input cell image

        Returns:
            List of processed image variants
        """
        variants = self._prepare_fast_variants(cell_image)
        variants.extend(self._prepare_fallback_variants(cell_image))
        return variants

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
