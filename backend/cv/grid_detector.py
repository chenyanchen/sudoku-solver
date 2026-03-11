"""Sudoku grid detection and perspective transformation."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .preprocessor import morphological_operations, preprocess


@dataclass
class GridDetectionDebug:
    """Intermediate results from the grid detection pipeline."""

    original: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    thresh: Optional[np.ndarray] = None
    morphed: Optional[np.ndarray] = None
    contours_image: Optional[np.ndarray] = None
    quad_image: Optional[np.ndarray] = None
    warped: Optional[np.ndarray] = None
    cells_montage: Optional[np.ndarray] = None
    corners: Optional[np.ndarray] = None
    error: Optional[str] = None


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

    Tries all quadrilateral contours (not just the largest) and picks the
    one whose warped result has the most regular grid-line spacing.  This
    handles decorative borders that are larger than the actual grid.

    Args:
        image: Input image (BGR format)

    Returns:
        Tuple of (warped_450x450_grid, ordered_corners_4x2_float32),
        or None if no valid grid is found.
    """
    try:
        processed = preprocess(image)
        thresh = morphological_operations(processed["thresh"])

        # RETR_LIST finds all contours including nested ones, so we can
        # skip decorative borders and find the actual grid inside.
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Collect all quadrilateral candidates, sorted by area (largest first).
        quads = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10000:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                quads.append((area, approx))

        quads.sort(key=lambda x: x[0], reverse=True)

        # Try each candidate: the first one with regular grid lines wins.
        fallback = None
        for _, contour in quads:
            corners = get_corner_points(contour)
            if corners is None:
                continue

            ordered = order_corners(corners)
            warped = perspective_transform(processed["original"], ordered, (450, 450))

            if not validate_grid(warped):
                continue

            gray_check = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            if _has_regular_grid_lines(gray_check):
                return warped, ordered

            if fallback is None:
                fallback = (warped, ordered)

        # No candidate passed regularity.  Try line-based refinement on the
        # best fallback (likely a decorative-border warp).
        if fallback is not None:
            refined = _try_refine_grid_bounds(
                fallback[0], processed["original"], fallback[1]
            )
            if refined is not None:
                return refined

        return fallback
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

        # RETR_LIST to find nested grids inside decorative borders.
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
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
        fallbacks: List[Tuple[np.ndarray, np.ndarray]] = []
        for _, contour in quads[: max_grids * 4]:
            corners = get_corner_points(contour)
            if corners is None:
                continue

            ordered = order_corners(corners)
            warped = perspective_transform(processed["original"], ordered, (450, 450))
            if not validate_grid(warped):
                continue

            gray_check = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            if _has_regular_grid_lines(gray_check):
                grids.append((warped, ordered))
                if len(grids) >= max_grids:
                    break
            elif len(fallbacks) < max_grids:
                fallbacks.append((warped, ordered))

        if grids:
            return grids

        # Try line-based refinement on fallbacks.
        for warped, ordered in fallbacks:
            refined = _try_refine_grid_bounds(warped, processed["original"], ordered)
            if refined is not None:
                grids.append(refined)
                if len(grids) >= max_grids:
                    break
        return grids if grids else fallbacks
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


def _find_line_peaks(projection: np.ndarray, min_prominence: float = 0.3) -> np.ndarray:
    """Find peaks in a 1D projection that correspond to grid lines.

    Groups consecutive above-threshold pixels and returns the centre of
    each group.

    Args:
        projection: 1D array of row/column sums from a binary line mask.
        min_prominence: Minimum peak height relative to max, as a fraction.

    Returns:
        Array of peak centre indices.
    """
    if projection.size == 0 or projection.max() == 0:
        return np.array([], dtype=int)

    threshold = projection.max() * min_prominence
    above = projection > threshold

    peaks: list[int] = []
    start: Optional[int] = None
    for i in range(len(above)):
        if above[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                peaks.append((start + i - 1) // 2)
                start = None
    if start is not None:
        peaks.append((start + len(above) - 1) // 2)

    return np.array(peaks, dtype=int)


def _detect_grid_lines(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect horizontal and vertical grid lines via morphological projection.

    Returns:
        (h_peaks, v_peaks, h_lines_mask) where peaks are 1-D arrays of row/col
        indices and h_lines_mask is the binary mask of horizontal lines.
    """
    h, w = gray.shape[:2]

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 5, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 5))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    h_proj = h_lines.sum(axis=1).astype(float)
    v_proj = v_lines.sum(axis=0).astype(float)

    return _find_line_peaks(h_proj), _find_line_peaks(v_proj), h_lines


def _direction_is_regular(
    peaks: np.ndarray, dim_size: int, max_cv: float = 0.25, min_span: float = 0.7
) -> bool:
    """Check if a set of peaks has regular spacing spanning most of the axis."""
    if len(peaks) < 8:
        return False
    gaps = np.diff(peaks).astype(float)
    if gaps.mean() == 0:
        return False
    if gaps.std() / gaps.mean() >= max_cv:
        return False
    span = float(peaks[-1] - peaks[0])
    return span >= dim_size * min_span


def _has_regular_grid_lines(
    gray: np.ndarray, min_lines: int = 8, max_cv: float = 0.25
) -> bool:
    """Check whether the warped image contains equally-spaced grid lines.

    A correctly warped 450x450 Sudoku grid has 10 horizontal + 10 vertical
    lines spaced ~50px apart.  An incorrectly warped decorative border will
    have lines clustered in the centre with irregular spacing.

    Two paths to pass:
    1. **Both** directions have ≥8 regular peaks spanning ≥70% of the image.
    2. **One** direction has ≥9 regular peaks AND average spacing is within
       15% of the expected cell size (image_dim / 9).  This handles grids
       whose thin lines are invisible in one direction due to border texture.

    Args:
        gray: Grayscale warped image (expected 450x450).
        min_lines: Minimum number of line peaks required in each direction.
        max_cv: Maximum coefficient of variation for the peak spacing.

    Returns:
        True if the image has a regular grid pattern.
    """
    h, w = gray.shape[:2]
    h_peaks, v_peaks, _ = _detect_grid_lines(gray)

    h_ok = _direction_is_regular(h_peaks, h, max_cv)
    v_ok = _direction_is_regular(v_peaks, w, max_cv)

    if h_ok and v_ok:
        return True

    # Single-direction fallback: accept if one direction has ≥9 peaks with
    # average spacing close to the expected cell size (~50 px for 450 px).
    expected = float(max(h, w)) / 9.0
    tol = 0.15  # ±15 %

    for peaks, ok in [(h_peaks, h_ok), (v_peaks, v_ok)]:
        if not ok or len(peaks) < 9:
            continue
        avg = float(np.diff(peaks).mean())
        if abs(avg - expected) / expected < tol:
            return True

    return False


def _try_refine_grid_bounds(
    warped: np.ndarray,
    original: np.ndarray,
    outer_corners: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Refine grid bounds when the warped image fails the regularity check.

    Uses the horizontal line mask to estimate the inner grid extent, maps
    the tighter corners back to the original image, and re-warps.  This
    handles decorative borders where the inner grid doesn't form a separate
    contour but its grid lines are visible in the initial warp.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    h, w = gray.shape[:2]

    h_peaks, v_peaks, h_lines_mask = _detect_grid_lines(gray)

    # Need enough horizontal peaks to estimate cell size.
    if len(h_peaks) < 8:
        return None

    # Select the best 10 consecutive peaks (lowest CV).
    best_h = _select_best_consecutive(h_peaks, 10)
    if best_h is None:
        return None

    cell_h = float(np.diff(best_h).mean())

    # Use V peaks for cell width, fall back to cell_h (square cells).
    if len(v_peaks) >= 2:
        cell_w = float(np.median(np.diff(v_peaks)))
    else:
        cell_w = cell_h

    # --- Y extent from best H peaks ---
    y_top = float(best_h[0])
    y_bot = float(best_h[-1])

    # --- X extent from H-line mask endpoints (per-row) ---
    # The grid may be trapezoidal in the warped image due to remaining
    # perspective distortion.  Use the top and bottom rows' endpoints to
    # build a quadrilateral (not a rectangle) for accurate correction.
    row_extents: list[Tuple[int, int, int]] = []  # (y, x_left, x_right)
    for y in best_h:
        row = h_lines_mask[int(y), :]
        nz = np.nonzero(row)[0]
        if len(nz) > 0:
            row_extents.append((int(y), int(nz[0]), int(nz[-1])))
    if len(row_extents) < 2:
        return None

    # Average the top-N and bottom-N rows for robustness.
    n_avg = min(3, len(row_extents) // 2)
    top_rows = row_extents[:n_avg]
    bot_rows = row_extents[-n_avg:]

    tl_x = float(np.mean([r[1] for r in top_rows]))
    tr_x = float(np.mean([r[2] for r in top_rows]))
    bl_x = float(np.mean([r[1] for r in bot_rows]))
    br_x = float(np.mean([r[2] for r in bot_rows]))

    # Sanity: inner grid must be meaningfully smaller than the warped image.
    top_w = tr_x - tl_x
    bot_w = br_x - bl_x
    avg_w = (top_w + bot_w) / 2.0
    grid_h = y_bot - y_top
    inner_area = avg_w * grid_h
    if inner_area < 0.25 * w * h or inner_area > 0.98 * w * h:
        return None

    # Build a quadrilateral (may be trapezoidal) that matches the actual
    # grid shape, preserving perspective information for accurate correction.
    inner_corners = np.array(
        [[tl_x, y_top], [tr_x, y_top], [br_x, y_bot], [bl_x, y_bot]],
        dtype=np.float32,
    )
    dst_pts = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32
    )
    M_inv = cv2.getPerspectiveTransform(dst_pts, outer_corners)
    mapped = cv2.perspectiveTransform(inner_corners.reshape(-1, 1, 2), M_inv)
    inner_in_orig = mapped.reshape(-1, 2).astype(np.float32)

    # Re-warp from the original image.
    refined = perspective_transform(original, inner_in_orig, (w, h))

    # Accept if the refined warp passes the regularity check.
    ref_gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY) if len(refined.shape) == 3 else refined
    if not _has_regular_grid_lines(ref_gray):
        return None

    return refined, inner_in_orig


def _select_best_consecutive(peaks: np.ndarray, n: int = 10) -> Optional[np.ndarray]:
    """Select the *n* consecutive peaks with the most regular spacing."""
    if len(peaks) < n:
        return None
    if len(peaks) == n:
        gaps = np.diff(peaks).astype(float)
        if gaps.mean() > 0 and gaps.std() / gaps.mean() < 0.25:
            return peaks
        return None

    best: Optional[np.ndarray] = None
    best_cv = float("inf")
    for start in range(len(peaks) - n + 1):
        subset = peaks[start : start + n]
        gaps = np.diff(subset).astype(float)
        if gaps.mean() == 0:
            continue
        cv = float(gaps.std() / gaps.mean())
        if cv < best_cv:
            best_cv = cv
            best = subset

    return best if best is not None and best_cv < 0.25 else None


def find_grid_with_debug(
    image: np.ndarray,
    reader: object = None,
) -> GridDetectionDebug:
    """Run the full grid detection pipeline and capture every intermediate step.

    *reader* is an optional ``CnnDigitReader`` instance used to annotate the
    cells montage with OCR predictions.
    """
    debug = GridDetectionDebug()

    try:
        debug.original = image.copy()

        processed = preprocess(image)
        debug.gray = processed["gray"]
        debug.thresh = processed["thresh"]

        morphed = morphological_operations(processed["thresh"])
        debug.morphed = morphed

        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            debug.error = "no contours found"
            return debug

        # Draw all contours on a copy of the original.
        contours_vis = debug.original.copy()
        cv2.drawContours(contours_vis, contours, -1, (0, 255, 0), 2)
        debug.contours_image = contours_vis

        grid_contour = find_largest_quadrilateral(contours)
        if grid_contour is None:
            debug.error = "no quadrilateral found"
            return debug

        corners = get_corner_points(grid_contour)
        if corners is None:
            debug.error = "could not extract corners"
            return debug

        ordered = order_corners(corners)
        debug.corners = ordered

        # Quad visualisation.
        quad_vis = debug.original.copy()
        pts = ordered.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(quad_vis, [pts], True, (0, 0, 255), 2)
        for pt in ordered:
            cv2.circle(quad_vis, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)
        debug.quad_image = quad_vis

        warped = perspective_transform(processed["original"], ordered, (450, 450))
        debug.warped = warped

        if not validate_grid(warped):
            debug.error = "grid validation failed"
            return debug

        # Build 9×9 cells montage.
        try:
            from backend.api.routes import prepare_grid_for_ocr
            from backend.cv.cell_extractor import extract_cells

            ocr_grid = prepare_grid_for_ocr(warped)
            cells = extract_cells(ocr_grid)

            cell_h, cell_w = cells[0].shape[:2] if cells else (50, 50)
            montage = np.ones((cell_h * 9, cell_w * 9), dtype=np.uint8) * 255
            for idx, cell in enumerate(cells[:81]):
                r, c = divmod(idx, 9)
                montage[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell

            # Convert to colour for annotation.
            montage_bgr = cv2.cvtColor(montage, cv2.COLOR_GRAY2BGR)

            if reader is not None and hasattr(reader, "recognize_grid_with_metadata"):
                _, metadata = reader.recognize_grid_with_metadata(cells, threshold=None)
                preds = metadata.get("cell_predictions", [])
                for pred in preds:
                    r_idx = int(pred["row"])
                    c_idx = int(pred["col"])
                    val = int(pred.get("value", 0))
                    conf = float(pred.get("confidence", 0.0))
                    if val != 0:
                        cx = c_idx * cell_w + cell_w // 2
                        cy = r_idx * cell_h + cell_h // 2
                        text = f"{val}:{conf:.0%}"
                        cv2.putText(
                            montage_bgr, text,
                            (cx - 15, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1,
                        )

            debug.cells_montage = montage_bgr
        except Exception:
            pass  # cells montage is best-effort

    except Exception as exc:
        debug.error = str(exc)

    return debug


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
