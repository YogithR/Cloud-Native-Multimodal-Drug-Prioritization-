"""Local MLflow helpers (Phase 9). File-backed store; no paid services."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.utils.config import load_yaml_config

_mlflow = None


def _get_mlflow():
    global _mlflow
    if _mlflow is None:
        import mlflow

        _mlflow = mlflow
    return _mlflow


def is_mlflow_enabled(mlflow_config_path: str = "configs/mlflow.yaml") -> bool:
    if os.environ.get("MLFLOW_DISABLE", "").lower() in ("1", "true", "yes"):
        return False
    cfg = load_yaml_config(mlflow_config_path).get("mlflow", {})
    return bool(cfg.get("enabled", True))


def configure_experiment(mlflow_config_path: str = "configs/mlflow.yaml") -> str:
    """Set tracking URI and experiment; return experiment name."""
    mlflow = _get_mlflow()
    cfg = load_yaml_config(mlflow_config_path)["mlflow"]
    uri = str(cfg["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    name = str(cfg["experiment_name"])
    mlflow.set_experiment(name)
    return name


def log_params_flat(params: dict[str, Any]) -> None:
    """Log params (values coerced to strings for MLflow compatibility)."""
    mlflow = _get_mlflow()
    flat = {k: str(v) for k, v in params.items()}
    mlflow.log_params(flat)


def log_sklearn_model(model: Any, artifact_path: str = "model") -> None:
    """Log sklearn model to the active run."""
    mlflow = _get_mlflow()
    mlflow.sklearn.log_model(model, artifact_path)


def log_file_artifact(local_path: str | Path, artifact_path: str | None = None) -> None:
    """Log a file as an artifact (e.g. joblib bundle path)."""
    mlflow = _get_mlflow()
    path = Path(local_path)
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
