"""Baseline model evaluation for Phase 4."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from src.utils.config import load_yaml_config
from src.utils.mlflow_tracking import configure_experiment, is_mlflow_enabled


def compute_metrics(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> dict[str, float]:
    """Compute locked Phase 4 metrics for classification."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate_split(
    split_path: str | Path,
    model: Any,
    feature_columns: list[str],
    threshold: float,
) -> dict[str, float]:
    """Evaluate a single split and return metric dictionary."""
    df = pd.read_csv(split_path)
    x = df[feature_columns]
    y_true = df["label"].astype(int)
    y_prob = pd.Series(model.predict_proba(x)[:, 1])
    return compute_metrics(y_true, y_prob, threshold)


def append_run_log(log_path: str | Path, row: dict[str, Any]) -> None:
    """Append evaluation summary row to a local CSV run log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_evaluation(
    config_path: str = "configs/model.yaml",
    model_key: str = "baseline",
) -> dict[str, Any]:
    """Run model evaluation on val/test splits and persist reports."""
    cfg = load_yaml_config(config_path)["model"]
    selected = cfg[model_key]
    threshold = float(selected["threshold"])
    artifact = joblib.load(selected["model_artifact_path"])
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    val_metrics = evaluate_split(
        selected["val_features_path"],
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
    )
    test_metrics = evaluate_split(
        selected["test_features_path"],
        model=model,
        feature_columns=feature_columns,
        threshold=threshold,
    )

    report = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "model_key": model_key,
        "threshold": threshold,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "model_artifact_path": selected["model_artifact_path"],
    }
    report_path = Path(selected["eval_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    log_row = {
        "run_timestamp_utc": report["run_timestamp_utc"],
        "model_key": model_key,
        "model_artifact_path": selected["model_artifact_path"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_f1": val_metrics["f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }
    append_run_log(selected["run_log_path"], log_row)

    if is_mlflow_enabled():
        configure_experiment()
        with mlflow.start_run(run_name=f"{model_key}_eval"):
            mlflow.set_tags({"model_key": model_key, "phase": "evaluation"})
            mlflow.log_metric("threshold", float(threshold))
            for split_name, metrics in report["metrics"].items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{split_name}_{metric_name}", float(value))

    return report


def run_graph_fusion_evaluation(config_path: str = "configs/model.yaml") -> dict[str, Any]:
    """Evaluate graph-fusion NN on val/test; persist JSON + CSV log + optional MLflow."""
    from src.models.graph_fusion_predict import (
        build_model_from_artifact,
        load_graph_fusion_artifact,
        predict_proba_positive_batch,
    )

    cfg = load_yaml_config(config_path)["model"]
    selected = cfg["graph_fusion"]
    threshold = float(selected["threshold"])
    artifact = load_graph_fusion_artifact(selected["model_artifact_path"])
    model = build_model_from_artifact(artifact)

    def _split_metrics(split_path: str | Path) -> dict[str, float]:
        df = pd.read_csv(split_path)
        y_true = df["label"].astype(int)
        y_prob = pd.Series(
            predict_proba_positive_batch(model, artifact, df),
        )
        return compute_metrics(y_true, y_prob, threshold)

    val_metrics = _split_metrics(selected["val_features_path"])
    test_metrics = _split_metrics(selected["test_features_path"])

    report = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "model_key": "graph_fusion",
        "threshold": threshold,
        "metrics": {"val": val_metrics, "test": test_metrics},
        "model_artifact_path": selected["model_artifact_path"],
        "architecture": "GCN_graph_branch_MLP_tabular_fusion",
    }
    report_path = Path(selected["eval_report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    log_row = {
        "run_timestamp_utc": report["run_timestamp_utc"],
        "model_key": "graph_fusion",
        "model_artifact_path": selected["model_artifact_path"],
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_f1": val_metrics["f1"],
        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
    }
    append_run_log(selected["run_log_path"], log_row)

    if is_mlflow_enabled():
        configure_experiment()
        with mlflow.start_run(run_name="graph_fusion_eval"):
            mlflow.set_tags({"model_key": "graph_fusion", "phase": "evaluation"})
            mlflow.log_metric("threshold", float(threshold))
            for split_name, metrics in report["metrics"].items():
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{split_name}_{metric_name}", float(value))

    return report


def build_comparison_report(config_path: str = "configs/model.yaml") -> dict[str, Any]:
    """Create baseline-vs-multimodal comparison report from metric files."""
    cfg = load_yaml_config(config_path)["model"]
    base_path = Path(cfg["baseline"]["eval_report_path"])
    mm_path = Path(cfg["multimodal"]["eval_report_path"])
    if not base_path.exists() or not mm_path.exists():
        raise FileNotFoundError("Both baseline and multimodal evaluation reports must exist first.")

    with base_path.open("r", encoding="utf-8") as file:
        baseline = json.load(file)
    with mm_path.open("r", encoding="utf-8") as file:
        multimodal = json.load(file)

    metric_names = ["roc_auc", "pr_auc", "f1", "balanced_accuracy"]
    comparison: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "baseline_report_path": str(base_path),
        "multimodal_report_path": str(mm_path),
        "deltas": {"val": {}, "test": {}},
    }
    for split in ["val", "test"]:
        for metric in metric_names:
            b = baseline["metrics"][split][metric]
            m = multimodal["metrics"][split][metric]
            comparison["deltas"][split][metric] = round(m - b, 6)

    comparison["baseline_metrics"] = baseline["metrics"]
    comparison["multimodal_metrics"] = multimodal["metrics"]

    output_path = Path(cfg["multimodal"]["comparison_report_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
    return comparison


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result, indent=2))
