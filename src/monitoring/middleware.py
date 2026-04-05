"""HTTP request logging middleware (Phase 11)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.monitoring.logging_config import get_logger, log_event

logger = get_logger("src.monitoring.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status, and duration for each request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start = time.perf_counter()
        path = request.url.path
        method = request.method
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            log_event(
                logger,
                "request_failed",
                method=method,
                path=path,
                duration_ms=round(duration_ms, 2),
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        log_event(
            logger,
            "request_completed",
            method=method,
            path=path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
