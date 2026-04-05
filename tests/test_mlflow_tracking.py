"""Tests for Phase 9 MLflow helpers."""

from pathlib import Path

import pytest

from src.utils.mlflow_tracking import is_mlflow_enabled


def test_is_mlflow_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_DISABLE", "1")
    assert is_mlflow_enabled() is False


def test_is_mlflow_enabled_when_env_cleared(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MLFLOW_DISABLE", raising=False)
    cfg_path = tmp_path / "mlflow.yaml"
    cfg_yaml = """mlflow:
  enabled: true
  tracking_uri: file:./mlruns
  experiment_name: test-exp
"""
    cfg_path.write_text(cfg_yaml)
    assert is_mlflow_enabled(str(cfg_path)) is True
