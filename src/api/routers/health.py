"""Health endpoint router."""

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.utils.config import load_yaml_config

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    serving = load_yaml_config("configs/serving.yaml")["serving"]
    return HealthResponse(
        status="ok",
        app_name=str(serving["app_name"]),
        version=str(serving["version"]),
    )
