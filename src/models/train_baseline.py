"""Baseline model training for Phase 4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.utils.config import load_yaml_config
from src.utils.mlflow_tracking import (
    configure_experiment,
    is_mlflow_enabled,
    log_file_artifact,
    log_params_flat,
    log_sklearn_model,
)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns from a feature table."""
    excluded = {"candidate_id", "smiles", "label"}
    return [col for col in df.columns if col not in excluded]


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load training split and return X, y, feature column names."""
    df = pd.read_csv(path)
    feature_cols = get_feature_columns(df)
    x_train = df[feature_cols]
    y_train = df["label"].astype(int)
    return x_train, y_train, feature_cols


def train_logistic_baseline(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    max_iter: int,
    solver: str,
    class_weight: str,
) -> LogisticRegression:
    """Train a deterministic logistic regression baseline model."""
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
    )
    model.fit(x_train, y_train)
    return model


def save_training_artifacts(
    model: LogisticRegression,
    model_path: str | Path,
    summary_path: str | Path,
    feature_columns: list[str],
    train_rows: int,
    model_config: dict[str, Any],
) -> None:
    """Save model artifact and train summary metadata."""
    model_path = Path(model_path)
    summary_path = Path(summary_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "model_config": model_config,
    }
    joblib.dump(artifact, model_path)

    summary = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "train_rows": train_rows,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "model_type": "LogisticRegression",
        "model_config": model_config,
        "model_artifact_path": str(model_path),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def run_training(config_path: str = "configs/model.yaml") -> dict[str, Any]:
    """Run baseline training and return summary payload."""
    cfg = load_yaml_config(config_path)["model"]
    baseline = cfg["baseline"]
    x_train, y_train, feature_columns = load_training_data(baseline["train_features_path"])

    model = train_logistic_baseline(
        x_train=x_train,
        y_train=y_train,
        random_state=int(cfg["random_state"]),
        max_iter=int(baseline["max_iter"]),
        solver=str(baseline["solver"]),
        class_weight=str(baseline["class_weight"]),
    )
    model_cfg = {
        "random_state": int(cfg["random_state"]),
        "max_iter": int(baseline["max_iter"]),
        "solver": str(baseline["solver"]),
        "class_weight": str(baseline["class_weight"]),
    }
    save_training_artifacts(
        model=model,
        model_path=baseline["model_artifact_path"],
        summary_path=baseline["train_summary_path"],
        feature_columns=feature_columns,
        train_rows=len(x_train),
        model_config=model_cfg,
    )
    if is_mlflow_enabled():
        configure_experiment()
        with mlflow.start_run(run_name="baseline_train"):
            mlflow.set_tags({"model_family": "baseline", "estimator": "LogisticRegression"})
            log_params_flat(
                {
                    **model_cfg,
                    "train_rows": len(x_train),
                    "feature_count": len(feature_columns),
                }
            )
            log_sklearn_model(model, "sklearn_model")
            log_file_artifact(baseline["train_summary_path"])

    return {
        "train_rows": len(x_train),
        "feature_count": len(feature_columns),
        "model_artifact_path": baseline["model_artifact_path"],
        "train_summary_path": baseline["train_summary_path"],
    }


if __name__ == "__main__":
    result = run_training()
    print(f"Baseline training complete: {result}")
