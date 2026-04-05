from pathlib import Path

import joblib
import pandas as pd

from src.models.evaluate import build_comparison_report, compute_metrics
from src.models.train_baseline import get_feature_columns, train_logistic_baseline
from src.models.train_multimodal import train_multimodal_fusion_model


def _sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [1, 2, 3, 4, 5, 6],
            "smiles": ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCF"],
            "label": [1, 0, 1, 0, 1, 0],
            "meta_smiles_length": [3, 3, 3, 4, 4, 3],
            "desc_mol_wt": [46.0, 44.0, 45.0, 64.5, 108.9, 48.0],
            "fp_000": [1, 0, 1, 0, 1, 0],
            "fp_001": [0, 1, 0, 1, 0, 1],
        }
    )


def test_train_logistic_baseline_and_save_artifact(tmp_path: Path) -> None:
    df = _sample_features()
    feature_cols = get_feature_columns(df)
    x = df[feature_cols]
    y = df["label"]
    model = train_logistic_baseline(
        x_train=x,
        y_train=y,
        random_state=42,
        max_iter=200,
        solver="liblinear",
        class_weight="balanced",
    )
    artifact_path = tmp_path / "baseline.joblib"
    joblib.dump({"model": model, "feature_columns": feature_cols}, artifact_path)
    loaded = joblib.load(artifact_path)
    assert "model" in loaded
    assert loaded["feature_columns"] == feature_cols


def test_compute_metrics_has_locked_keys() -> None:
    y_true = pd.Series([0, 1, 0, 1])
    y_prob = pd.Series([0.1, 0.9, 0.2, 0.8])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert set(metrics.keys()) == {"roc_auc", "pr_auc", "f1", "balanced_accuracy"}


def test_train_multimodal_fusion_model_produces_probabilities() -> None:
    df = _sample_features()
    feature_cols = get_feature_columns(df)
    x = df[feature_cols]
    y = df["label"]
    model = train_multimodal_fusion_model(
        x_train=x,
        y_train=y,
        random_state=42,
        n_estimators=20,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
    )
    probs = model.predict_proba(x)[:, 1]
    assert len(probs) == len(df)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_build_comparison_report_generates_deltas(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline_metrics.json"
    multimodal_report = tmp_path / "multimodal_metrics.json"
    output_report = tmp_path / "comparison.json"
    baseline_report.write_text(
        """
{
  "metrics": {
    "val": {"roc_auc": 0.90, "pr_auc": 0.91, "f1": 0.80, "balanced_accuracy": 0.81},
    "test": {"roc_auc": 0.92, "pr_auc": 0.93, "f1": 0.82, "balanced_accuracy": 0.83}
  }
}
""".strip(),
        encoding="utf-8",
    )
    multimodal_report.write_text(
        """
{
  "metrics": {
    "val": {"roc_auc": 0.95, "pr_auc": 0.96, "f1": 0.84, "balanced_accuracy": 0.86},
    "test": {"roc_auc": 0.96, "pr_auc": 0.97, "f1": 0.86, "balanced_accuracy": 0.88}
  }
}
""".strip(),
        encoding="utf-8",
    )
    config = tmp_path / "model.yaml"
    config.write_text(
        f"""
model:
  baseline:
    eval_report_path: {baseline_report.as_posix()}
  multimodal:
    eval_report_path: {multimodal_report.as_posix()}
    comparison_report_path: {output_report.as_posix()}
""".strip(),
        encoding="utf-8",
    )
    result = build_comparison_report(str(config))
    assert result["deltas"]["val"]["roc_auc"] == 0.05
    assert output_report.exists()
