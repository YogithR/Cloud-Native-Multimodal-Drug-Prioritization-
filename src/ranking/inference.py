"""Unified sklearn vs graph-fusion inference for ranking and API."""

from __future__ import annotations

from typing import Any, Literal

import joblib
import pandas as pd

from src.utils.config import load_yaml_config

InferenceKind = Literal["sklearn", "graph_fusion"]


def load_model_and_columns(config_path: str, model_key: str) -> tuple[Any, list[str]]:
    """Load sklearn model artifact from Phase 4/5 layout."""
    cfg = load_yaml_config(config_path)["model"]
    path = cfg[model_key]["model_artifact_path"]
    artifact = joblib.load(path)
    return artifact["model"], artifact["feature_columns"]


def predict_prob_positive(
    model: Any,
    feature_columns: list[str],
    df: pd.DataFrame,
) -> pd.Series:
    """P(y=1) for binary sklearn classifier."""
    x = df[feature_columns]
    return pd.Series(model.predict_proba(x)[:, 1], index=df.index)


def load_inference_bundle(
    config_path: str,
    model_key: str,
) -> tuple[InferenceKind, dict[str, Any]]:
    """
    Load model bundle for ranking/API.

    Returns (kind, bundle) where bundle is:
    - sklearn: {"model", "feature_columns"}
    - graph_fusion: {"artifact", "torch_model", "feature_columns"}
    """
    cfg = load_yaml_config(config_path)["model"]
    if model_key == "graph_fusion":
        from src.models.graph_fusion_predict import (
            build_model_from_artifact,
            load_graph_fusion_artifact,
        )

        path = cfg["graph_fusion"]["model_artifact_path"]
        artifact = load_graph_fusion_artifact(path)
        model = build_model_from_artifact(artifact)
        model.eval()
        return (
            "graph_fusion",
            {
                "artifact": artifact,
                "torch_model": model,
                "feature_columns": artifact["feature_columns"],
            },
        )
    model, feature_columns = load_model_and_columns(config_path, model_key)
    return "sklearn", {"model": model, "feature_columns": feature_columns}


def predict_prob_positive_unified(
    kind: InferenceKind,
    bundle: dict[str, Any],
    df: pd.DataFrame,
) -> pd.Series:
    """P(y=1) aligned with df index."""
    if kind == "sklearn":
        return predict_prob_positive(bundle["model"], bundle["feature_columns"], df)
    import torch

    from src.models.graph_fusion_predict import predict_proba_positive_batch

    probs = predict_proba_positive_batch(
        bundle["torch_model"],
        bundle["artifact"],
        df,
        device=torch.device("cpu"),
    )
    return pd.Series(probs, index=df.index)
