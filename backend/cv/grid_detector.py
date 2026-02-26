"""Sudoku grid detection and perspective transformation."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .preprocessor import morphological_operations, preprocess


def find_grid(image: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Find and extract the Sudoku grid from an image.

    Args:
        image: Input image (BGR format)
        debug: If True, return debug information

    Returns:
        Warped 450x450px grid image, or None if not found
    """
    result = find_grid_with_corners(image)
    if result is None:
        return None
    warped, _ = result
    return warped


def find_grid_with_corners(
    image: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Find and extract the Sudoku grid and return screen-space corners.

    Args:
        image: Input image (BGR format)

    Returns:
        Tuple of (warped_450x450_grid, ordered_corners_4x2_float32),
        or None if no valid grid is found.
    """
    try:
        processed = preprocess(image)

        # Apply morphological operations to strengthen grid lines.
        thresh = morphological_operations(processed["thresh"])

        # Find contours.
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find the largest contour that could be the grid.
        grid_contour = find_largest_quadrilateral(contours)
        if grid_contour is None:
            return None

        corners = get_corner_points(grid_contour)
        if corners is None:
            return None

        ordered = order_corners(corners)

        # Perspective transform to 450x450 (50px per cell).
        warped = perspective_transform(processed["original"], ordered, (450, 450))

        if not validate_grid(warped):
            return None

        return warped, ordered
    except Exception:
        return None


def find_grids_with_corners(
    image: np.ndarray,
    max_grids: int = 8,
    min_area_ratio: float = 0.02,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Find multiple potential Sudoku grids with corners.

    Args:
        image: Input image
        max_grids: Maximum number of grids to return
        min_area_ratio: Minimum candidate area relative to input image area

    Returns:
        List of (warped_grid, ordered_corners)
    """
    try:
        processed = preprocess(image)
        thresh = morphological_operations(processed["thresh"])

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        image_h, image_w = processed["original"].shape[:2]
        image_area = float(image_h * image_w)
        min_area = max(10000.0, image_area * float(min_area_ratio))

        # Find all quadrilateral contours and sort by area.
        quads = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approximation) == 4:
                quads.append((area, approximation))

        quads.sort(key=lambda x: x[0], reverse=True)

        grids: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, contour in quads[: max_grids * 2]:
            corners = get_corner_points(contour)
            if corners is None:
                continue

            ordered = order_corners(corners)
            warped = perspective_transform(processed["original"], ordered, (450, 450))
            if not validate_grid(warped):
                continue

            grids.append((warped, ordered))
            if len(grids) >= max_grids:
                break

        return grids
    except Exception:
        return []


def find_largest_quadrilateral(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Find the largest 4-sided contour.

    Args:
        contours: List of contours

    Returns:
        Largest quadrilateral contour or None
    """
    best_contour = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip contours that are too small.
        if area < 10000:
            continue

        # Approximate contour to a polygon.
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # We want a quadrilateral (4 vertices).
        if len(approximation) == 4 and area > best_area:
            best_area = area
            best_contour = approximation

    return best_contour


def get_corner_points(contour: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract corner points from a quadrilateral contour.

    Args:
        contour: 4-point contour

    Returns:
        Array of 4 corner points
    """
    if contour is None or len(contour) != 4:
        return None

    # Reshape to (4, 2).
    corners = contour.reshape(4, 2).astype(np.float32)
    return corners


def order_corners(corners: np.ndarray) -> np.ndarray:
    """
    Order corner points: top-left, top-right, bottom-right, bottom-left.

    Args:
        corners: Array of 4 corner points

    Returns:
        Ordered array of corners
    """
    # Sort by x coordinate.
    sorted_by_x = corners[np.argsort(corners[:, 0])]

    # Left points and right points.
    left_points = sorted_by_x[:2]
    right_points = sorted_by_x[2:]

    # Sort left by y (top-left first).
    left_points = left_points[np.argsort(left_points[:, 1])]
    top_left = left_points[0]
    bottom_left = left_points[1]

    # Sort right by y (top-right first).
    right_points = right_points[np.argsort(right_points[:, 1])]
    top_right = right_points[0]
    bottom_right = right_points[1]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def perspective_transform(
    image: np.ndarray, corners: np.ndarray, output_size: Tuple[int, int] = (450, 450)
) -> np.ndarray:
    """
    Apply perspective transform to get a top-down view of the grid.

    Args:
        image: Input image
        corners: Ordered corner points
        output_size: Size of output image (width, height)

    Returns:
        Warped image
    """
    width, height = output_size

    # Destination points (top-left, top-right, bottom-right, bottom-left).
    dst_points = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    # Calculate perspective transform matrix.
    matrix = cv2.getPerspectiveTransform(corners, dst_points)

    # Apply transformation.
    warped = cv2.warpPerspective(image, matrix, output_size)

    return warped


def validate_grid(image: np.ndarray) -> bool:
    """
    Validate that the extracted image looks like a Sudoku grid.

    Args:
        image: Extracted grid image

    Returns:
        True if likely a valid Sudoku grid
    """
    if image is None or image.shape[0] < 100 or image.shape[1] < 100:
        return False

    # Check for grid lines by looking for horizontal and vertical edges.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Count horizontal and vertical lines using Hough lines.
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10
    )

    if lines is None:
        return False

    # A Sudoku grid should have enough visible lines.
    return len(lines) >= 15


def find_grids_multiple(image: np.ndarray, max_grids: int = 5) -> List[np.ndarray]:
    """
    Find multiple potential Sudoku grids in an image.

    Args:
        image: Input image
        max_grids: Maximum number of grids to return

    Returns:
        List of warped grid images
    """
    try:
        processed = preprocess(image)
        thresh = morphological_operations(processed["thresh"])

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Find all quadrilateral contours and sort by area.
        quads = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                continue

            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approximation) == 4:
                quads.append((area, approximation))

        # Sort by area (largest first) and take top candidates.
        quads.sort(key=lambda x: x[0], reverse=True)
        quads = quads[:max_grids]

        grids = []
        for _, contour in quads:
            corners = get_corner_points(contour)
            if corners is not None:
                corners = order_corners(corners)
                warped = perspective_transform(
                    processed["original"], corners, (450, 450)
                )
                if validate_grid(warped):
                    grids.append(warped)

        return grids

    except Exception:
        return []
