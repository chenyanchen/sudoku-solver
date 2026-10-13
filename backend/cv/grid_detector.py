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


def find_grid_with_corners(
    image: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Find and extract the Sudoku grid and return screen-space corners.

    Tries all quadrilateral contours (not just the largest) and picks the
    one whose warped result has the most regular grid-line spacing.  This
    handles decorative borders that are larger than the actual grid.

    Returns:
        Tuple of (warped_450x450_grid, ordered_corners_4x2_float32),
        or None if no valid grid is found.
    """
    grids = _find_grids(image, max_grids=1, min_area=10000.0, max_quads=None)
    return grids[0] if grids else None


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
    image_h, image_w = image.shape[:2]
    min_area = max(10000.0, float(image_h * image_w) * float(min_area_ratio))
    return _find_grids(image, max_grids, min_area, max_quads=max_grids * 4)


def _find_grids(
    image: np.ndarray,
    max_grids: int,
    min_area: float,
    max_quads: Optional[int],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Warp quadrilateral candidates (largest first); prefer regular grids.

    Candidates that fail the regularity check are kept as fallbacks and
    get one chance at line-based refinement (decorative borders).
    """
    thresh = morphological_operations(preprocess(image)["thresh"])
    # RETR_LIST finds nested contours, so we can skip decorative borders
    # and find the actual grid inside.
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    grids: List[Tuple[np.ndarray, np.ndarray]] = []
    fallbacks: List[Tuple[np.ndarray, np.ndarray]] = []
    for ordered in _quad_corners(contours, min_area)[:max_quads]:
        warped = perspective_transform(image, ordered, (450, 450))
        if not validate_grid(warped):
            continue

        if has_regular_grid_lines(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)):
            grids.append((warped, ordered))
            if len(grids) >= max_grids:
                break
        elif len(fallbacks) < max_grids:
            fallbacks.append((warped, ordered))

    if grids:
        return grids

    for warped, ordered in fallbacks:
        refined = _try_refine_grid_bounds(warped, image, ordered)
        if refined is not None:
            grids.append(refined)
            if len(grids) >= max_grids:
                break
    return grids if grids else fallbacks


def _quad_corners(contours, min_area: float) -> List[np.ndarray]:
    """Ordered corners of every 4-vertex contour >= *min_area*, largest first."""
    quads = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            quads.append((area, order_corners(approx.reshape(4, 2).astype(np.float32))))
    quads.sort(key=lambda x: x[0], reverse=True)
    return [corners for _, corners in quads]


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


def has_regular_grid_lines(gray: np.ndarray, max_cv: float = 0.25) -> bool:
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
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
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

    # --- Y extent from best H peaks ---
    y_top = float(best_h[0])
    y_bot = float(best_h[-1])

    # --- X extent from H-line mask endpoints (per-row) ---
    # Fit a line through (y, x_left) and (y, x_right) to extrapolate
    # corners at y_top/y_bot while preserving trapezoidal shape.
    row_extents: list[Tuple[int, int, int]] = []  # (y, x_left, x_right)
    for y in best_h:
        row = h_lines_mask[int(y), :]
        nz = np.nonzero(row)[0]
        if len(nz) > 0:
            row_extents.append((int(y), int(nz[0]), int(nz[-1])))
    if len(row_extents) < 2:
        return None

    # Remove endpoint outliers separately for x_left and x_right.
    # Decorative borders can truncate individual line endpoints severely,
    # causing x_left to be too far right or x_right too far left.
    med_xl = float(np.median([r[1] for r in row_extents]))
    med_xr = float(np.median([r[2] for r in row_extents]))
    row_extents = [
        r for r in row_extents
        if r[1] <= med_xl + cell_h and r[2] >= med_xr - cell_h
    ]
    if len(row_extents) < 2:
        return None

    ys = np.array([r[0] for r in row_extents], dtype=float)
    x_lefts = np.array([r[1] for r in row_extents], dtype=float)
    x_rights = np.array([r[2] for r in row_extents], dtype=float)

    left_fit = np.polyfit(ys, x_lefts, 1)
    right_fit = np.polyfit(ys, x_rights, 1)

    tl_x = float(np.polyval(left_fit, y_top))
    tr_x = float(np.polyval(right_fit, y_top))
    bl_x = float(np.polyval(left_fit, y_bot))
    br_x = float(np.polyval(right_fit, y_bot))

    # H-line endpoints may not reach the true grid boundary when the
    # decorative border obscures the intersection with vertical lines.
    # Use V peaks to widen each corner independently if a vertical grid
    # line exists just outside the endpoint estimate (within one cell).
    if len(v_peaks) >= 2:
        v_gaps = np.diff(v_peaks).astype(float)
        single_gaps = v_gaps[v_gaps < cell_h * 1.8]
        if len(single_gaps) > 0:
            v_cell = float(np.median(single_gaps))
            v_left = float(v_peaks[0])
            v_right_est = v_left + 9 * v_cell
            # Expand left corners independently.
            if v_left < tl_x and (tl_x - v_left) < v_cell:
                tl_x = v_left
            if v_left < bl_x and (bl_x - v_left) < v_cell:
                bl_x = v_left
            # Expand right corners independently.
            if v_right_est > tr_x and (v_right_est - tr_x) < v_cell:
                tr_x = v_right_est
            if v_right_est > br_x and (v_right_est - br_x) < v_cell:
                br_x = v_right_est

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
    if not has_regular_grid_lines(cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)):
        return None

    return refined, inner_in_orig


def _select_best_consecutive(peaks: np.ndarray, n: int = 10) -> Optional[np.ndarray]:
    """Select the *n* consecutive peaks with the most regular spacing."""
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

    quads = _quad_corners(contours, 10000.0)
    if not quads:
        debug.error = "no quadrilateral found"
        return debug

    ordered = quads[0]
    debug.corners = ordered

    # Quad visualisation.
    quad_vis = debug.original.copy()
    pts = ordered.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(quad_vis, [pts], True, (0, 0, 255), 2)
    for pt in ordered:
        cv2.circle(quad_vis, (int(pt[0]), int(pt[1])), 8, (255, 0, 0), -1)
    debug.quad_image = quad_vis

    warped = perspective_transform(image, ordered, (450, 450))
    debug.warped = warped

    if not validate_grid(warped):
        debug.error = "grid validation failed"
        return debug

    # Build 9×9 cells montage.
    from backend.ocr.pipeline import prepare_grid_for_ocr
    from backend.cv.cell_extractor import extract_cells

    ocr_grid = prepare_grid_for_ocr(warped)
    cells = extract_cells(ocr_grid)

    cell_h, cell_w = cells[0].shape[:2]
    montage_bgr = np.full((cell_h * 9, cell_w * 9, 3), 255, dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, 9)
        montage_bgr[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = cell

    if reader is not None:
        _, metadata = reader.recognize_grid_with_metadata(cells, threshold=None)
        for pred in metadata["cell_predictions"]:
            val = int(pred["value"])
            if val != 0:
                cx = int(pred["col"]) * cell_w + cell_w // 2
                cy = int(pred["row"]) * cell_h + cell_h // 2
                cv2.putText(
                    montage_bgr, f"{val}:{float(pred['confidence']):.0%}",
                    (cx - 15, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1,
                )

    debug.cells_montage = montage_bgr
    return debug
