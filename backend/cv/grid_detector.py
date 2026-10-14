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
    snapped to the grid-line lattice around them.
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
        elif len(fallbacks) < max(max_grids, _MAX_FALLBACKS):
            fallbacks.append((warped, ordered))

    if grids:
        return grids

    # An irregular quad is often a decorative frame around the grid, or a
    # single 3x3 box when the grid's outline merges with its surroundings.
    for _, ordered in fallbacks:
        refined = snap_to_lattice(image, ordered, cells=9) or snap_to_lattice(
            image, ordered, cells=3,
        )
        if refined is not None:
            grids.append(refined)
            if len(grids) >= max_grids:
                break
    return grids if grids else fallbacks[:max_grids]


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


def _detect_grid_lines(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect horizontal and vertical grid lines via morphological projection.

    Returns:
        (h_peaks, v_peaks): 1-D arrays of row/col indices.
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

    return _find_line_peaks(h_proj), _find_line_peaks(v_proj)


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
    h_peaks, v_peaks = _detect_grid_lines(gray)

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


_MAX_FALLBACKS = 8
_SNAP_PX = 30  # canvas pixels per cell


def snap_to_lattice(
    image: np.ndarray, corners: np.ndarray, cells: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Find the 9x9 grid around a quad assumed to span `cells` x `cells` cells.

    A whole grid (cells=9) is searched with 1.5 cells of margin. A 3x3 box
    (cells=3) is first re-fitted to its own four lines, since extrapolating
    six cells amplifies its corner error, then searched with six cells of
    margin and accepted only if it sits on the lattice's box boundaries.
    The quad only has to be roughly right: lines are located, not assumed.
    """
    if cells < 9:
        box = _fit_lattice(image, corners, cells, margin=0.5, lines=cells + 1)
        if box is None:
            return None
        corners, margin = box[0], 9 - cells
    else:
        margin = 1.5
    found = _fit_lattice(image, corners, cells, margin, lines=10)
    if found is None:
        return None
    snapped, offsets = found
    if cells < 9 and any(abs(o - 3 * round(o / 3)) > 0.3 for o in offsets):
        return None
    warped = perspective_transform(image, snapped, (450, 450))
    if not has_regular_grid_lines(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)):
        return None
    return warped, snapped


def _fit_lattice(
    image: np.ndarray, corners: np.ndarray, cells: int, margin: float, lines: int,
) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
    """Fit `lines` equally spaced lines per axis around a quad of `cells` cells.

    Lines are measured in two strips across the quad's middle, which every
    grid line crosses, so each boundary line gets its own tilt and the fit
    can correct the quad's perspective, not just its position and scale.

    Returns (the fitted lattice's corners in image coordinates, the quad's
    offset from the lattice's first line in cells per axis), or None.
    """
    px = _SNAP_PX
    size = int(round((cells + 2 * margin) * px))
    src = np.float32([[0, 0], [cells, 0], [cells, cells], [0, cells]])
    cell_to_image = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    canvas_to_image = cell_to_image @ np.float32(
        [[1 / px, 0, -margin], [0, 1 / px, -margin], [0, 0, 1]]
    )
    canvas = cv2.warpPerspective(
        image, canvas_to_image, (size, size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_AREA,
    )
    binary = cv2.adaptiveThreshold(
        cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10,
    )
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (px, 1)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, px)))

    a, b, c = (int(round((margin + cells * f) * px)) for f in (0.1, 0.5, 0.9))
    centres = ((a + b) / 2, (b + c) / 2)
    min_hits = lines - 2 if lines > 4 else lines
    rows, cols = [], []
    for lo, hi in ((a, b), (b, c)):
        rows.append(_fit_lines(_coverage_peaks(h_lines[:, lo:hi].mean(axis=1) / 255, 0.5), px, lines, min_hits))
        cols.append(_fit_lines(_coverage_peaks(v_lines[lo:hi, :].mean(axis=0) / 255, 0.5), px, lines, min_hits))
    if None in rows or None in cols:
        return None
    # Both strips must have found the same lines, not lattices a line apart.
    if max(abs(rows[0][k] - rows[1][k]) for k in (0, 1)) > px / 2:
        return None
    if max(abs(cols[0][k] - cols[1][k]) for k in (0, 1)) > px / 2:
        return None

    def h_line(k: int) -> np.ndarray:
        return np.cross([centres[0], rows[0][k], 1.0], [centres[1], rows[1][k], 1.0])

    def v_line(k: int) -> np.ndarray:
        return np.cross([cols[0][k], centres[0], 1.0], [cols[1][k], centres[1], 1.0])

    points = [np.cross(v_line(x), h_line(y)) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    rect = np.float32([p[:2] / p[2] for p in points])
    fitted = cv2.perspectiveTransform(rect.reshape(-1, 1, 2), canvas_to_image).reshape(4, 2)
    offsets = tuple(
        (margin * px - (fit[0][0] + fit[1][0]) / 2)
        / ((fit[0][1] + fit[1][1] - fit[0][0] - fit[1][0]) / 2 / (lines - 1))
        for fit in (cols, rows)
    )
    return fitted.astype(np.float32), offsets


def _coverage_peaks(profile: np.ndarray, threshold: float) -> np.ndarray:
    """Centres of the runs where `profile` exceeds `threshold`."""
    above = np.concatenate([[0], (profile > threshold).astype(int), [0]])
    edges = np.diff(above)
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    return ((starts + ends - 1) / 2).astype(float)


def _fit_lines(
    peaks: np.ndarray, spacing: float, count: int, min_hits: int,
) -> Optional[Tuple[float, float]]:
    """(first, last) of `count` equally spaced lines (spacing ±25%) fitted to `peaks`.

    Up to count - min_hits lines may be missing (faint or hidden by glare).
    Each pair of peaks proposes a spacing and phase; the proposal matching
    the most peaks, then with the smallest residual, wins.
    """
    best: Optional[Tuple[Tuple[int, float], Tuple[float, float]]] = None
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            gaps = round((peaks[j] - peaks[i]) / spacing)
            if not 1 <= gaps < count:
                continue
            step = (peaks[j] - peaks[i]) / gaps
            if abs(step - spacing) > 0.25 * spacing:
                continue
            for first in range(count - gaps):
                lines = peaks[i] + (np.arange(count) - first) * step
                dist = np.abs(peaks[:, None] - lines[None, :])
                hit = np.nonzero(dist.min(axis=0) < 0.15 * step)[0]
                if len(hit) < min_hits:
                    continue
                matched = peaks[dist[:, hit].argmin(axis=0)]
                slope, start = np.polyfit(hit, matched, 1)
                residual = float(np.abs(start + slope * hit - matched).mean())
                key = (len(hit), -residual)
                if best is None or key > best[0]:
                    best = (key, (float(start), float(start + (count - 1) * slope)))
    return None if best is None else best[1]


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
