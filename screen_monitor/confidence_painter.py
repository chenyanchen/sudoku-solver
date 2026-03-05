"""OCR confidence heatmap painter for the debug overlay."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PyQt6.QtGui import QPainter


def confidence_color(conf: float) -> tuple[int, int, int]:
    """Map a confidence value to an (R, G, B) colour.

    - >= 0.95  → green  (60, 200, 60)
    - 0.80–0.95 → yellow (220, 200, 40)
    - < 0.80   → red    (220, 60, 60)
    """
    if conf >= 0.95:
        return (60, 200, 60)
    if conf >= 0.80:
        return (220, 200, 40)
    return (220, 60, 60)


def paint_confidence_grid(
    painter: QPainter,
    cells: list[dict[str, Any]],
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    """Draw a small coloured rectangle for each cell based on OCR confidence."""
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QColor

    painter.setPen(Qt.PenStyle.NoPen)

    for cell in cells:
        x = int(cell["x"]) - offset_x
        y = int(cell["y"]) - offset_y
        size = int(cell.get("size", 20))
        conf = float(cell.get("confidence", 0.0))
        r, g, b = confidence_color(conf)
        painter.setBrush(QColor(r, g, b, 100))
        painter.drawRect(x - size // 2, y - size // 2, size, size)
