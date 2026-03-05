"""Tests for TelemetryRing push / latest / recent."""

from screen_monitor.telemetry import FrameTelemetry, TelemetryRing


def test_empty_ring():
    ring = TelemetryRing(capacity=5)
    assert len(ring) == 0
    assert ring.latest() is None
    assert ring.recent(3) == []


def test_push_and_latest():
    ring = TelemetryRing(capacity=5)
    ring.push(FrameTelemetry(fps=1.0))
    ring.push(FrameTelemetry(fps=2.0))
    assert len(ring) == 2
    assert ring.latest().fps == 2.0


def test_recent_returns_chronological():
    ring = TelemetryRing(capacity=5)
    for i in range(4):
        ring.push(FrameTelemetry(fps=float(i)))
    result = ring.recent(3)
    assert [e.fps for e in result] == [1.0, 2.0, 3.0]


def test_ring_wraps_correctly():
    ring = TelemetryRing(capacity=3)
    for i in range(5):
        ring.push(FrameTelemetry(fps=float(i)))
    assert len(ring) == 3
    assert ring.latest().fps == 4.0
    result = ring.recent(3)
    assert [e.fps for e in result] == [2.0, 3.0, 4.0]


def test_recent_more_than_available():
    ring = TelemetryRing(capacity=10)
    ring.push(FrameTelemetry(fps=1.0))
    ring.push(FrameTelemetry(fps=2.0))
    result = ring.recent(100)
    assert len(result) == 2
    assert [e.fps for e in result] == [1.0, 2.0]


def test_capacity_one():
    ring = TelemetryRing(capacity=1)
    ring.push(FrameTelemetry(fps=1.0))
    ring.push(FrameTelemetry(fps=2.0))
    assert len(ring) == 1
    assert ring.latest().fps == 2.0
    assert ring.recent(5) == [ring.latest()]
