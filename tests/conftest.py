"""Pytest defaults: disable MLflow side effects unless a test opts in."""

import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE", "1")


@pytest.fixture(scope="module")
def torch_cpu():
    """Skip graph-fusion tests when PyTorch is missing or broken (partial install)."""
    try:
        import torch
        import torch.nn as nn

        torch.device("cpu")
        nn.Linear(1, 1)
        return torch
    except Exception as exc:
        pytest.skip(f"PyTorch not usable for graph tests: {exc}")
