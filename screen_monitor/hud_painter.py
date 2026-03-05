"""Debug HUD overlay painter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtGui import QPainter

    from .telemetry import TelemetryRing


def paint_debug_hud(painter: QPainter, ring: TelemetryRing) -> None:
    """Draw a semi-transparent debug HUD in the top-left corner."""
    from PyQt6.QtCore import QRect, Qt
    from PyQt6.QtGui import QColor, QFont

    entry = ring.latest()
    if entry is None:
        return

    # Compute average FPS from recent entries.
    recent = ring.recent(30)
    avg_fps = sum(e.fps for e in recent) / max(1, len(recent))

    lines = [
        f"FPS: {avg_fps:5.1f}  (cur {entry.fps:5.1f})",
        f"Detect: {entry.detect_ms:6.1f} ms  {'HIT' if entry.detect_cache_hit else 'MISS'}",
        f"Solve:  {entry.solve_ms:6.1f} ms  {'HIT' if entry.solve_cache_hit else 'MISS'}",
        f"Stable: {entry.stable_count}  State: {entry.state}",
    ]

    font = QFont("Menlo", 10)
    painter.setFont(font)

    line_height = 16
    padding = 6
    width = 300
    height = len(lines) * line_height + padding * 2

    # Background.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(0, 0, 0, 160))
    painter.drawRect(QRect(8, 8, width, height))

    # Text.
    painter.setPen(QColor(255, 255, 255, 230))
    for i, line in enumerate(lines):
        painter.drawText(
            8 + padding,
            8 + padding + (i + 1) * line_height - 2,
            line,
        )
