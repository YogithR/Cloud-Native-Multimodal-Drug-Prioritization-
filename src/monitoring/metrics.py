"""Prometheus metrics for FastAPI (Phase 11)."""

from __future__ import annotations

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator


def setup_prometheus_metrics(app: FastAPI) -> None:
    """
    Expose /metrics in Prometheus text format.

    Includes default process and HTTP metrics (latency histograms, etc.).
    """
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics", "/health"],
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)
