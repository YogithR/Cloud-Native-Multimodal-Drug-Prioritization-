"""Structured-style logging for the API (Phase 11)."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any


def configure_logging(
    level: str | None = None,
    *,
    service_name: str = "drug-prioritization-api",
) -> None:
    """
    Configure root logging once. Uses key=value lines for grep-friendly logs.

    Set LOG_LEVEL=DEBUG|INFO|WARNING (default INFO).
    """
    log_level = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, log_level, logging.INFO)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | " f"service={service_name} | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the app namespace."""
    return logging.getLogger(name)


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Log a single line with key=value pairs (structured for operators)."""
    if not fields:
        logger.info(message)
        return
    parts = [f"{k}={v!r}" for k, v in fields.items()]
    logger.info("%s | %s", message, " ".join(parts))
