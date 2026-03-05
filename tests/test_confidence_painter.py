"""Tests for confidence_color boundary values."""

from screen_monitor.confidence_painter import confidence_color


def test_high_confidence_green():
    assert confidence_color(1.0) == (60, 200, 60)
    assert confidence_color(0.95) == (60, 200, 60)
    assert confidence_color(0.99) == (60, 200, 60)


def test_medium_confidence_yellow():
    assert confidence_color(0.94) == (220, 200, 40)
    assert confidence_color(0.80) == (220, 200, 40)
    assert confidence_color(0.87) == (220, 200, 40)


def test_low_confidence_red():
    assert confidence_color(0.79) == (220, 60, 60)
    assert confidence_color(0.0) == (220, 60, 60)
    assert confidence_color(0.50) == (220, 60, 60)
