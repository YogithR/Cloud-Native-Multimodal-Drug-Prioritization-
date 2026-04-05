"""FastAPI application entrypoint."""

from fastapi import FastAPI

from src.api.routers.health import router as health_router
from src.api.routers.predict import router as predict_router
from src.api.routers.rank import router as rank_router
from src.monitoring.logging_config import configure_logging
from src.monitoring.metrics import setup_prometheus_metrics
from src.monitoring.middleware import RequestLoggingMiddleware

configure_logging()

app = FastAPI(
    title="Drug Prioritization API",
    version="0.1.0",
    description="Local-first API scaffold for drug candidate prioritization.",
)
app.add_middleware(RequestLoggingMiddleware)
setup_prometheus_metrics(app)

app.include_router(health_router)
app.include_router(predict_router)
app.include_router(rank_router)
