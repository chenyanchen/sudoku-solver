"""PyQt6 transparent click-through overlay window."""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import QObject, QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

from .telemetry import TelemetryRing


class OverlaySignals(QObject):
    update_signal = pyqtSignal(list)
    clear_signal = pyqtSignal()
    error_signal = pyqtSignal()  # FIX #2: worker crash → quit app
    confidence_signal = pyqtSignal(list)


class OverlayWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._cells: list[dict[str, Any]] = []
        self._confidence_cells: list[dict[str, Any]] = []
        self._font = QFont("Helvetica Neue", 20, QFont.Weight.Bold)
        self._fm = QFontMetrics(self._font)  # FIX #3: proper centering
        self._telemetry_ring: Optional[TelemetryRing] = None
        self._debug_hud: bool = False

        window_flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
        )
        # macOS: Tool windows hide when the app loses focus — don't use Tool flag.
        if hasattr(Qt.WindowType, "WindowDoesNotAcceptFocus"):
            window_flags |= Qt.WindowType.WindowDoesNotAcceptFocus
        self.setWindowFlags(window_flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if hasattr(Qt.WidgetAttribute, "WA_ShowWithoutActivating"):
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.virtualGeometry())

    def set_debug_hud(self, enabled: bool) -> None:
        self._debug_hud = enabled

    def set_telemetry_ring(self, ring: TelemetryRing) -> None:
        self._telemetry_ring = ring

    def set_cells(self, cells: list[dict[str, Any]]) -> None:
        self._cells = cells
        self.update()

    def clear_cells(self) -> None:
        if not self._cells and not self._confidence_cells:
            return
        self._cells = []
        self._confidence_cells = []
        self.update()

    def set_confidence_cells(self, cells: list[dict[str, Any]]) -> None:
        self._confidence_cells = cells
        self.update()

    def paintEvent(self, _event: Any) -> None:  # pragma: no cover
        has_content = (
            self._cells
            or (self._debug_hud and self._telemetry_ring)
            or (self._debug_hud and self._confidence_cells)
        )
        if not has_content:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Confidence heatmap (under hint digits).
        if self._debug_hud and self._confidence_cells:
            from .confidence_painter import paint_confidence_grid

            paint_confidence_grid(painter, self._confidence_cells)

        # Hint digits.
        if self._cells:
            painter.setFont(self._font)

            bg_color = QColor(30, 80, 200, 150)
            fg_color = QColor(255, 255, 255, 240)
            radius = 13

            for cell in self._cells:
                x = int(cell["x"])
                y = int(cell["y"])
                text = str(cell["digit"])

                # Background circle.
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(bg_color)
                painter.drawEllipse(QPoint(x, y), radius, radius)

                # FIX #3: center text using QFontMetrics instead of hardcoded offset.
                text_rect: QRect = self._fm.boundingRect(text)
                painter.setPen(fg_color)
                painter.drawText(
                    x - text_rect.width() // 2,
                    y + self._fm.ascent() // 2 - 1,
                    text,
                )

        # Debug HUD (on top of everything).
        if self._debug_hud and self._telemetry_ring:
            from .hud_painter import paint_debug_hud

            paint_debug_hud(painter, self._telemetry_ring)
