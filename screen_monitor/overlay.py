"""PyQt6 transparent click-through overlay window."""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
from typing import Any, Optional

from PyQt6.QtCore import QObject, QPoint, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PyQt6.QtWidgets import QApplication, QWidget

from .telemetry import TelemetryRing

LOGGER = logging.getLogger("screen_monitor")


class OverlaySignals(QObject):
    update_signal = pyqtSignal(list)
    clear_signal = pyqtSignal()
    error_signal = pyqtSignal()  # FIX #2: worker crash → quit app
    confidence_signal = pyqtSignal(list)
    geometry_signal = pyqtSignal(int, int, int, int)


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

    def show(self) -> None:
        super().show()
        self._exclude_from_capture()

    def _exclude_from_capture(self) -> None:
        """Set NSWindow.sharingType = NSWindowSharingNone on macOS.

        This prevents mss (which uses CGWindowListCreateImage) from
        capturing the overlay, avoiding a feedback loop where the
        overlay's own pixels interfere with grid detection.
        """
        if sys.platform != "darwin":
            return
        try:
            lib = ctypes.util.find_library("objc")
            if not lib:
                return
            objc = ctypes.cdll.LoadLibrary(lib)

            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            sel_window = objc.sel_registerName(b"window")
            sel_set_sharing = objc.sel_registerName(b"setSharingType:")

            # [NSView window] → NSWindow*
            send = objc.objc_msgSend
            send.restype = ctypes.c_void_p
            send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            ns_view = int(self.winId())
            ns_window = send(ns_view, sel_window)
            if not ns_window:
                return

            # [NSWindow setSharingType: NSWindowSharingNone(0)]
            send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            send(ns_window, sel_set_sharing, 0)
            LOGGER.debug("overlay window excluded from screen capture")
        except Exception:
            LOGGER.debug("failed to exclude overlay from capture", exc_info=True)

    def move_to_screen(self, x: int, y: int, w: int, h: int) -> None:
        """Move overlay to cover the screen at (x, y, w, h)."""
        geo = self.geometry()
        if geo.x() != x or geo.y() != y or geo.width() != w or geo.height() != h:
            self.setGeometry(x, y, w, h)
            LOGGER.debug("overlay moved to screen at (%d,%d) %dx%d", x, y, w, h)

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

        # Convert global logical coords to widget-local coords.
        # The overlay spans virtualGeometry, so painter (0,0) =
        # virtualGeometry.topLeft() in global coords.
        ox = self.x()
        oy = self.y()

        # Confidence heatmap (under hint digits).
        if self._debug_hud and self._confidence_cells:
            from .confidence_painter import paint_confidence_grid

            paint_confidence_grid(painter, self._confidence_cells, ox, oy)

        # Hint digits.
        if self._cells:
            painter.setFont(self._font)

            bg_color = QColor(30, 80, 200, 150)
            fg_color = QColor(255, 255, 255, 240)
            radius = 13

            for cell in self._cells:
                x = int(cell["x"]) - ox
                y = int(cell["y"]) - oy
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
