"""Multimodal fusion model training for Phase 5."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.train_baseline import get_feature_columns
from src.utils.config import load_yaml_config
from src.utils.mlflow_tracking import (
    configure_experiment,
    is_mlflow_enabled,
    log_file_artifact,
    log_params_flat,
    log_sklearn_model,
)


def _build_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    metadata = [c for c in feature_columns if c.startswith("meta_")]
    descriptors = [c for c in feature_columns if c.startswith("desc_")]
    fingerprints = [c for c in feature_columns if c.startswith("fp_")]
    return {
        "metadata_feature_columns": metadata,
        "molecular_descriptor_columns": descriptors,
        "molecular_fingerprint_columns": fingerprints,
    }


def train_multimodal_fusion_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
) -> RandomForestClassifier:
    """Train deterministic CPU-friendly fusion model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train)
    return model


def run_multimodal_training(config_path: str = "configs/model.yaml") -> dict[str, Any]:
    """Run multimodal training and persist artifacts."""
    cfg = load_yaml_config(config_path)["model"]
    mm = cfg["multimodal"]
    train_df = pd.read_csv(mm["train_features_path"])
    feature_columns = get_feature_columns(train_df)
    x_train = train_df[feature_columns]
    y_train = train_df["label"].astype(int)

    model = train_multimodal_fusion_model(
        x_train=x_train,
        y_train=y_train,
        random_state=int(cfg["random_state"]),
        n_estimators=int(mm["n_estimators"]),
        max_depth=int(mm["max_depth"]),
        min_samples_split=int(mm["min_samples_split"]),
        min_samples_leaf=int(mm["min_samples_leaf"]),
    )

    model_path = Path(mm["model_artifact_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    train_summary_path = Path(mm["train_summary_path"])
    train_summary_path.parent.mkdir(parents=True, exist_ok=True)

    feature_groups = _build_feature_groups(feature_columns)
    model_config = {
        "random_state": int(cfg["random_state"]),
        "n_estimators": int(mm["n_estimators"]),
        "max_depth": int(mm["max_depth"]),
        "min_samples_split": int(mm["min_samples_split"]),
        "min_samples_leaf": int(mm["min_samples_leaf"]),
    }
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "feature_groups": feature_groups,
        "model_config": model_config,
    }
    joblib.dump(artifact, model_path)

    summary = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "model_type": "RandomForestClassifier",
        "train_rows": len(train_df),
        "feature_count": len(feature_columns),
        "feature_groups": feature_groups,
        "model_config": model_config,
        "model_artifact_path": str(model_path),
    }
    with train_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if is_mlflow_enabled():
        configure_experiment()
        with mlflow.start_run(run_name="multimodal_train"):
            mlflow.set_tags({"model_family": "multimodal", "estimator": "RandomForestClassifier"})
            log_params_flat(
                {
                    **model_config,
                    "train_rows": len(train_df),
                    "feature_count": len(feature_columns),
                }
            )
            log_sklearn_model(model, "sklearn_model")
            log_file_artifact(mm["train_summary_path"])

    return summary


if __name__ == "__main__":
    result = run_multimodal_training()
    print(f"Multimodal training complete: {result}")
