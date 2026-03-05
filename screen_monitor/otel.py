"""OpenTelemetry metrics for the screen monitor pipeline.

All metrics are forwarded from ``TelemetryRing.push()`` via a single
callback — there are no scattered ``record_*`` calls elsewhere in the
codebase.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .telemetry import FrameTelemetry, TelemetryRing

LOGGER = logging.getLogger("screen_monitor.otel")


def setup_metrics(ring: TelemetryRing, port: int = 9092) -> None:
    """Initialise OTel metrics and wire them into *ring*."""
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from opentelemetry.metrics import set_meter_provider
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.resources import Resource
        from prometheus_client import start_http_server
    except ImportError:
        LOGGER.warning("OTel/Prometheus packages not available, metrics disabled")
        return

    start_http_server(port)
    LOGGER.info("Prometheus metrics endpoint started on :%d", port)

    resource = Resource.create({"service.name": "sudoku-screen-monitor"})
    reader = PrometheusMetricReader()
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    set_meter_provider(provider)

    meter = provider.get_meter("screen_monitor", "1.0.0")

    frame_counter = meter.create_counter(
        "sudoku_frame_total",
        description="Total frames processed by the capture loop",
    )
    step_duration = meter.create_histogram(
        "sudoku_step_duration_ms",
        description="Duration of each pipeline step in milliseconds",
        unit="ms",
    )
    cache_counter = meter.create_counter(
        "sudoku_cache_total",
        description="Cache operations by type and result",
    )
    givens_hist = meter.create_histogram(
        "sudoku_givens",
        description="Number of given cells in detected puzzles",
    )

    def _forward(entry: FrameTelemetry) -> None:
        frame_counter.add(1, {"state": entry.state})

        if entry.capture_ms > 0:
            step_duration.record(entry.capture_ms, {"step": "capture"})
        if entry.detect_ms > 0:
            step_duration.record(entry.detect_ms, {"step": "detect"})
        if entry.ocr_ms > 0:
            step_duration.record(entry.ocr_ms, {"step": "ocr"})
        if entry.solve_ms > 0:
            step_duration.record(entry.solve_ms, {"step": "solve"})
        if entry.render_ms > 0:
            step_duration.record(entry.render_ms, {"step": "render"})

        if entry.detect_cache_hit:
            cache_counter.add(1, {"cache": "detect", "result": "hit"})
        if entry.solve_cache_hit:
            cache_counter.add(1, {"cache": "solve", "result": "hit"})

        if entry.givens > 0:
            givens_hist.record(entry.givens)

    ring.set_otel_forwarder(_forward)
