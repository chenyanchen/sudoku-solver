"""Adaptive cell extraction using 16 keypoints (4x4 block intersections).

Two-level warp: 16 keypoints define 9 block quadrilaterals, each warped
independently to 150x150, then split into 3x3 cells of 50x50.
This handles decorative grids where block separators have uneven widths.
"""

from __future__ import annotations

import cv2
import numpy as np

_BLOCK_WARP_SIZE = 150
_CELL_SIZE = 50  # 150 / 3
_BORDER_SKIP = 1


def extract_cells_adaptive(
    image: np.ndarray,
    keypoints_16: np.ndarray,
    cell_output_size: int = 48,
) -> list[np.ndarray]:
    """Extract 81 cells from an image using 16 keypoints.

    Args:
        image: Original image (any size, BGR or grayscale).
        keypoints_16: Shape (16, 2), YOLO-detected keypoints in image coords.
                      Arranged as 4x4 grid (row-major): indices 0-3 = top row,
                      4-7 = second row, 8-11 = third row, 12-15 = bottom row.
        cell_output_size: Output cell size (default 48 to match CNN input).

    Returns:
        81 cell images in row-major order (top-left to bottom-right).
    """
    kpts = keypoints_16.reshape(4, 4, 2).astype(np.float32)

    dst_block = np.array([
        [0, 0],
        [_BLOCK_WARP_SIZE - 1, 0],
        [_BLOCK_WARP_SIZE - 1, _BLOCK_WARP_SIZE - 1],
        [0, _BLOCK_WARP_SIZE - 1],
    ], dtype=np.float32)

    cells: list[np.ndarray] = []

    for block_row in range(3):
        for block_col in range(3):
            src_quad = np.array([
                kpts[block_row, block_col],          # TL
                kpts[block_row, block_col + 1],      # TR
                kpts[block_row + 1, block_col + 1],  # BR
                kpts[block_row + 1, block_col],      # BL
            ], dtype=np.float32)

            matrix = cv2.getPerspectiveTransform(src_quad, dst_block)
            block_img = cv2.warpPerspective(image, matrix, (_BLOCK_WARP_SIZE, _BLOCK_WARP_SIZE))

            for cell_row in range(3):
                for cell_col in range(3):
                    x = cell_col * _CELL_SIZE
                    y = cell_row * _CELL_SIZE
                    cell = block_img[
                        y + _BORDER_SKIP : y + _CELL_SIZE - _BORDER_SKIP,
                        x + _BORDER_SKIP : x + _CELL_SIZE - _BORDER_SKIP,
                    ]
                    if cell_output_size != cell.shape[0]:
                        cell = cv2.resize(cell, (cell_output_size, cell_output_size))
                    cells.append(cell)

    return cells
