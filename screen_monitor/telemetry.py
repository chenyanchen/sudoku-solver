"""Frame telemetry data and ring buffer for the debug HUD."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameTelemetry:
    """Telemetry snapshot for a single capture-loop iteration."""

    fps: float = 0.0
    detect_ms: float = 0.0
    solve_ms: float = 0.0
    detect_cache_hit: bool = False
    solve_cache_hit: bool = False
    stable_count: int = 0
    state: str = "idle"  # idle | active | stable | lost


class TelemetryRing:
    """Fixed-capacity ring buffer for FrameTelemetry entries.

    Single-thread write (worker) + Qt-thread read (paint); GIL provides
    sufficient synchronisation for this use-case.
    """

    def __init__(self, capacity: int = 120) -> None:
        self._capacity = max(1, int(capacity))
        self._buf: list[FrameTelemetry] = []
        self._head: int = 0  # next write position (when full)

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, entry: FrameTelemetry) -> None:
        if len(self._buf) < self._capacity:
            self._buf.append(entry)
        else:
            self._buf[self._head] = entry
            self._head = (self._head + 1) % self._capacity

    def latest(self) -> Optional[FrameTelemetry]:
        if not self._buf:
            return None
        if len(self._buf) < self._capacity:
            return self._buf[-1]
        return self._buf[(self._head - 1) % self._capacity]

    def recent(self, n: int = 10) -> list[FrameTelemetry]:
        """Return the *n* most recent entries in chronological order."""
        if not self._buf:
            return []
        n = min(n, len(self._buf))
        if len(self._buf) < self._capacity:
            return list(self._buf[-n:])
        result: list[FrameTelemetry] = []
        start = (self._head - n) % self._capacity
        for i in range(n):
            result.append(self._buf[(start + i) % self._capacity])
        return result

    def __len__(self) -> int:
        return len(self._buf)
