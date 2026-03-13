"""CameraWidget: 2x2 panel layout — video, detected grid, OCR, solution."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from . import ProcessingResult

# Styling constants.
_BG = QColor(30, 30, 30)
_PANEL_BG = QColor(40, 40, 40)
_BORDER = QColor(60, 60, 60)
_TITLE_COLOR = QColor(160, 160, 160)
_GIVEN_COLOR = QColor(220, 220, 220)
_SOLVED_COLOR = QColor(80, 160, 255)
_GRID_LINE = QColor(100, 100, 100)
_GRID_LINE_THICK = QColor(180, 180, 180)
_OUTLINE_COLOR = QColor(0, 220, 0, 200)
_HINT_BG = QColor(30, 80, 200, 150)
_HINT_FG = QColor(255, 255, 255, 240)

_GAP = 8          # pixels between panels
_TITLE_H = 24     # title bar height
_TITLE_FONT = QFont("Helvetica Neue", 11)
_PANEL_TITLES = ["Camera", "Detected Grid", "Recognition", "Solution"]


class CameraWidget(QWidget):
    """2x2 panel widget: camera | warped grid | OCR result | solution."""

    def __init__(self, debug: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Sudoku Camera")
        self.resize(1000, 760)
        self.setMinimumSize(600, 450)
        self._debug = debug

        # Live camera frame (always updates).
        self._frame_rgb: Optional[np.ndarray] = None
        self._frame_h = 0
        self._frame_w = 0

        # Current detection result (for camera overlay).
        self._current = ProcessingResult()

        # Sticky results: keep last successful data for panels 2-4.
        self._sticky_warped: Optional[np.ndarray] = None   # BGR 450x450
        self._sticky_original_grid: Optional[list[list[int]]] = None
        self._sticky_solved_grid: Optional[list[list[int]]] = None
        self._sticky_corners: Optional[np.ndarray] = None
        self._sticky_hints: Optional[list[dict[str, Any]]] = None

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        """Store current camera frame (BGR) and schedule repaint."""
        self._frame_rgb = frame_bgr[:, :, ::-1].copy()
        self._frame_h, self._frame_w = frame_bgr.shape[:2]
        self.update()

    def set_result(self, result: ProcessingResult) -> None:
        """Update with latest processing result; sticky panels only update on success."""
        self._current = result

        # Update sticky warped whenever we have a detected grid.
        if result.warped is not None:
            self._sticky_warped = result.warped
            self._sticky_corners = result.corners
        # Update sticky grids only when we have a full solve.
        if result.original_grid is not None:
            self._sticky_original_grid = result.original_grid
        if result.solved_grid is not None:
            self._sticky_solved_grid = result.solved_grid
        if result.hints is not None:
            self._sticky_hints = result.hints

    # ---- Layout helpers ----

    def _panel_rects(self) -> list[QRect]:
        """Compute the 4 panel rectangles in 2x2 grid."""
        ww = self.width()
        wh = self.height()
        pw = (ww - 3 * _GAP) // 2
        ph = (wh - 3 * _GAP) // 2
        return [
            QRect(_GAP, _GAP, pw, ph),                         # top-left
            QRect(_GAP * 2 + pw, _GAP, pw, ph),                # top-right
            QRect(_GAP, _GAP * 2 + ph, pw, ph),                # bottom-left
            QRect(_GAP * 2 + pw, _GAP * 2 + ph, pw, ph),      # bottom-right
        ]

    # ---- Paint ----

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Background.
        painter.fillRect(self.rect(), _BG)

        panels = self._panel_rects()

        # Panel 0: Camera feed + overlay.
        self._paint_camera_panel(painter, panels[0])

        # Panel 1: Detected grid (warped 450x450).
        self._paint_warped_panel(painter, panels[1])

        # Panel 2: OCR recognition grid.
        self._paint_grid_panel(
            painter, panels[2], _PANEL_TITLES[2], self._sticky_original_grid, False,
        )

        # Panel 3: Solution grid.
        self._paint_grid_panel(
            painter, panels[3], _PANEL_TITLES[3], self._sticky_solved_grid, True,
            original_grid=self._sticky_original_grid,
        )

        painter.end()

    def _paint_panel_frame(self, painter: QPainter, rect: QRect, title: str) -> QRect:
        """Draw panel background + title, return content rect."""
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_PANEL_BG)
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(_TITLE_COLOR)
        painter.setFont(_TITLE_FONT)
        painter.drawText(
            rect.x() + 8, rect.y() + _TITLE_H - 6, title,
        )

        return QRect(
            rect.x() + 2, rect.y() + _TITLE_H,
            rect.width() - 4, rect.height() - _TITLE_H - 2,
        )

    def _paint_camera_panel(self, painter: QPainter, rect: QRect) -> None:
        content = self._paint_panel_frame(painter, rect, _PANEL_TITLES[0])

        if self._frame_rgb is None:
            return

        fw, fh = self._frame_w, self._frame_h
        scale = min(content.width() / fw, content.height() / fh)
        disp_w = int(fw * scale)
        disp_h = int(fh * scale)
        ox = content.x() + (content.width() - disp_w) // 2
        oy = content.y() + (content.height() - disp_h) // 2

        qimg = QImage(
            self._frame_rgb.data, fw, fh,
            fw * 3, QImage.Format.Format_RGB888,
        )
        painter.drawImage(QRect(ox, oy, disp_w, disp_h), qimg)

        result = self._current

        # Draw grid outline on camera.
        if result.corners is not None:
            pen = QPen(_OUTLINE_COLOR, 2)
            painter.setPen(pen)
            pts = [
                QPoint(int(result.corners[i, 0] * scale) + ox,
                       int(result.corners[i, 1] * scale) + oy)
                for i in range(4)
            ]
            for i in range(4):
                painter.drawLine(pts[i], pts[(i + 1) % 4])

        # Draw hint digits on camera.
        if self._sticky_hints and self._sticky_corners is not None:
            cell_size = self._estimate_cell_size(self._sticky_corners, scale)
            font_size = max(6, int(cell_size * 0.45))
            radius = max(3, int(cell_size * 0.24))

            font = QFont("Helvetica Neue", font_size, QFont.Weight.Bold)
            painter.setFont(font)
            fm = QFontMetrics(font)

            for hint in self._sticky_hints:
                x = int(hint["x"] * scale) + ox
                y = int(hint["y"] * scale) + oy
                text = str(hint["digit"])

                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(_HINT_BG)
                painter.drawEllipse(QPoint(x, y), radius, radius)

                text_rect = fm.boundingRect(text)
                painter.setPen(_HINT_FG)
                painter.drawText(
                    x - text_rect.width() // 2,
                    y + fm.ascent() // 2 - 1,
                    text,
                )

        # Debug: givens count.
        if self._debug and result.corners is not None:
            painter.setPen(QColor(0, 255, 0))
            painter.setFont(QFont("Helvetica Neue", 10))
            painter.drawText(ox + 4, oy + 14, f"givens: {result.givens}")

    def _paint_warped_panel(self, painter: QPainter, rect: QRect) -> None:
        content = self._paint_panel_frame(painter, rect, _PANEL_TITLES[1])

        warped = self._sticky_warped
        if warped is None:
            return

        rgb = warped[:, :, ::-1].copy()
        h, w = rgb.shape[:2]
        scale = min(content.width() / w, content.height() / h)
        dw = int(w * scale)
        dh = int(h * scale)
        ox = content.x() + (content.width() - dw) // 2
        oy = content.y() + (content.height() - dh) // 2

        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        painter.drawImage(QRect(ox, oy, dw, dh), qimg)

    def _paint_grid_panel(
        self,
        painter: QPainter,
        rect: QRect,
        title: str,
        grid: Optional[list[list[int]]],
        is_solution: bool,
        original_grid: Optional[list[list[int]]] = None,
    ) -> None:
        content = self._paint_panel_frame(painter, rect, title)

        if grid is None:
            return

        # Compute cell size to fit in content area (square grid).
        side = min(content.width(), content.height()) - 4
        cell = side // 9
        grid_side = cell * 9
        gx = content.x() + (content.width() - grid_side) // 2
        gy = content.y() + (content.height() - grid_side) // 2

        # Draw grid background.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(50, 50, 50))
        painter.drawRect(gx, gy, grid_side, grid_side)

        # Draw cells.
        font_size = max(8, int(cell * 0.55))
        font = QFont("Helvetica Neue", font_size, QFont.Weight.Bold)
        painter.setFont(font)
        fm = QFontMetrics(font)

        for row in range(9):
            for col in range(9):
                cx = gx + col * cell
                cy = gy + row * cell
                val = grid[row][col]

                # Cell background.
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(25, 25, 25))
                painter.drawRect(cx + 1, cy + 1, cell - 1, cell - 1)

                if val == 0:
                    continue

                # Determine if this is a given or solved cell.
                is_given = True
                if is_solution and original_grid is not None:
                    is_given = original_grid[row][col] != 0

                color = _GIVEN_COLOR if is_given else _SOLVED_COLOR
                painter.setPen(color)
                text = str(val)
                tw = fm.horizontalAdvance(text)
                tx = cx + (cell - tw) // 2
                ty = cy + (cell + fm.ascent()) // 2 - 2
                painter.drawText(tx, ty, text)

        # Draw grid lines.
        for i in range(10):
            thick = (i % 3 == 0)
            pen = QPen(_GRID_LINE_THICK if thick else _GRID_LINE, 2 if thick else 1)
            painter.setPen(pen)
            # Horizontal.
            painter.drawLine(gx, gy + i * cell, gx + grid_side, gy + i * cell)
            # Vertical.
            painter.drawLine(gx + i * cell, gy, gx + i * cell, gy + grid_side)

    @staticmethod
    def _estimate_cell_size(corners: np.ndarray, scale: float) -> float:
        xs = corners[:, 0] * scale
        ys = corners[:, 1] * scale
        return min(float(xs.max() - xs.min()), float(ys.max() - ys.min())) / 9.0
