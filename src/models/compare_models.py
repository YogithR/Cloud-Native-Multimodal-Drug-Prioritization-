"""Three-way comparison: baseline vs RF multimodal vs graph fusion (metrics + ranking + errors)."""

from __future__ import annotations

import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.ranking.score import PenaltyConfig, compute_priority_score
from src.utils.config import load_yaml_config


def _pred_sklearn(df: pd.DataFrame, artifact_path: str) -> np.ndarray:
    bundle = joblib.load(artifact_path)
    model = bundle["model"]
    cols = bundle["feature_columns"]
    return model.predict_proba(df[cols])[:, 1]


def _pred_graph_fusion(df: pd.DataFrame, artifact_path: str) -> np.ndarray:
    from src.models.graph_fusion_predict import (
        build_model_from_artifact,
        load_graph_fusion_artifact,
        predict_proba_positive_batch,
    )

    artifact = load_graph_fusion_artifact(artifact_path)
    model = build_model_from_artifact(artifact)
    return predict_proba_positive_batch(model, artifact, df)


def _priority_series(
    df: pd.DataFrame,
    probs: np.ndarray,
    positive_class_is_good: bool,
    penalty_cfg: PenaltyConfig,
) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.float64)
    for i, (_, row) in enumerate(df.iterrows()):
        p = float(probs[i])
        desc = {
            "desc_logp": row.get("desc_logp"),
            "desc_mol_wt": row.get("desc_mol_wt"),
            "desc_tpsa": row.get("desc_tpsa"),
        }
        priority, _ = compute_priority_score(
            prob_positive=p,
            positive_class_is_good=positive_class_is_good,
            penalty_cfg=penalty_cfg,
            descriptors=desc,
        )
        out[i] = priority
    return out


def _error_analysis(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    name: str,
) -> dict[str, Any]:
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in cm.ravel())
    return {
        "model": name,
        "threshold": threshold,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation via rank Pearson (pandas only)."""
    s1 = pd.Series(a).rank(method="average")
    s2 = pd.Series(b).rank(method="average")
    val = s1.corr(s2, method="pearson")
    return float(val) if val == val else 0.0


def run_three_way_comparison(config_path: str = "configs/model.yaml") -> dict[str, Any]:
    cfg = load_yaml_config(config_path)["model"]
    ranking_cfg = load_yaml_config("configs/ranking.yaml")["ranking"]
    pen = ranking_cfg["penalties"]
    penalty_cfg = PenaltyConfig(
        high_logp_threshold=float(pen["high_logp_threshold"]),
        high_logp_penalty=float(pen["high_logp_penalty"]),
        high_mw_threshold=float(pen["high_mw_threshold"]),
        high_mw_penalty=float(pen["high_mw_penalty"]),
        high_tpsa_threshold=float(pen["high_tpsa_threshold"]),
        high_tpsa_penalty=float(pen["high_tpsa_penalty"]),
    )
    positive_is_good = bool(ranking_cfg["positive_class_is_good"])

    test_path = cfg["baseline"]["test_features_path"]
    df = pd.read_csv(test_path)
    y_true = df["label"].astype(int).to_numpy()

    paths = {
        "baseline": cfg["baseline"]["model_artifact_path"],
        "multimodal": cfg["multimodal"]["model_artifact_path"],
        "graph_fusion": cfg["graph_fusion"]["model_artifact_path"],
    }
    thresholds = {
        "baseline": float(cfg["baseline"]["threshold"]),
        "multimodal": float(cfg["multimodal"]["threshold"]),
        "graph_fusion": float(cfg["graph_fusion"]["threshold"]),
    }

    probs = {
        "baseline": _pred_sklearn(df, paths["baseline"]),
        "multimodal": _pred_sklearn(df, paths["multimodal"]),
        "graph_fusion": _pred_graph_fusion(df, paths["graph_fusion"]),
    }

    metric_reports = {}
    for key in ["baseline", "multimodal", "graph_fusion"]:
        p = Path(cfg[key]["eval_report_path"])
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                metric_reports[key] = json.load(f)["metrics"]
        else:
            metric_reports[key] = {"val": {}, "test": {}, "note": "eval report missing"}

    rank_corr: dict[str, float] = {}
    keys = ["baseline", "multimodal", "graph_fusion"]
    for a, b in itertools.combinations(keys, 2):
        ra = pd.Series(-probs[a]).rank(method="average").to_numpy()
        rb = pd.Series(-probs[b]).rank(method="average").to_numpy()
        rank_corr[f"{a}_vs_{b}_prob_rank_spearman"] = _spearman_rho(ra, rb)

    priorities = {
        k: _priority_series(df, probs[k], positive_is_good, penalty_cfg) for k in keys
    }
    for a, b in itertools.combinations(keys, 2):
        ra = pd.Series(-priorities[a]).rank(method="average").to_numpy()
        rb = pd.Series(-priorities[b]).rank(method="average").to_numpy()
        rank_corr[f"{a}_vs_{b}_priority_rank_spearman"] = _spearman_rho(ra, rb)

    errors = [_error_analysis(y_true, probs[k], thresholds[k], k) for k in keys]

    baseline_wrong = (y_true != (probs["baseline"] >= thresholds["baseline"]).astype(int))
    gf_right = y_true == (probs["graph_fusion"] >= thresholds["graph_fusion"]).astype(int)
    mm_wrong = y_true != (probs["multimodal"] >= thresholds["multimodal"]).astype(int)
    same_pred_mask = (
        (
            (probs["baseline"] >= thresholds["baseline"]).astype(int)
            == (probs["multimodal"] >= thresholds["multimodal"]).astype(int)
        )
        & (
            (probs["multimodal"] >= thresholds["multimodal"]).astype(int)
            == (probs["graph_fusion"] >= thresholds["graph_fusion"]).astype(int)
        )
    )
    error_analysis = {
        "rows_on_test": len(df),
        "baseline_errors_where_graph_fusion_correct": int((baseline_wrong & gf_right).sum()),
        "multimodal_errors_where_graph_fusion_correct": int((mm_wrong & gf_right).sum()),
        "all_three_same_prediction": int(same_pred_mask.sum()),
    }

    out: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "test_features_path": test_path,
        "metrics_from_reports": metric_reports,
        "ranking_correlation": rank_corr,
        "per_model_confusion_style_counts": errors,
        "error_analysis": error_analysis,
        "notes": [
            "Spearman on rankdata(-prob) approximates sorting by descending predicted P(y=1).",
            "Priority Spearman uses the same penalty config as configs/ranking.yaml.",
        ],
    }

    out_path = Path(cfg["graph_fusion"]["three_way_comparison_report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(run_three_way_comparison(), indent=2))
