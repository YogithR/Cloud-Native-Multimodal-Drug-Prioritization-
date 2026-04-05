"""Load model, score candidates, assign ranks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.ranking.explain import build_reason_codes
from src.ranking.inference import (
    load_inference_bundle,
    load_model_and_columns,
    predict_prob_positive,
    predict_prob_positive_unified,
)
from src.ranking.score import PenaltyConfig, compute_priority_score
from src.utils.config import load_yaml_config


def rank_candidates_frame(
    df: pd.DataFrame,
    prob_positive: pd.Series,
    positive_class_is_good: bool,
    penalty_cfg: PenaltyConfig,
) -> pd.DataFrame:
    """Add priority_score, rank, reason_codes, prediction_confidence per row."""
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        p = float(prob_positive.loc[idx])
        confidence = abs(p - 0.5) * 2.0
        desc = {
            "desc_logp": row.get("desc_logp"),
            "desc_mol_wt": row.get("desc_mol_wt"),
            "desc_tpsa": row.get("desc_tpsa"),
        }
        priority, pen = compute_priority_score(
            prob_positive=p,
            positive_class_is_good=positive_class_is_good,
            penalty_cfg=penalty_cfg,
            descriptors=desc,
        )
        reasons = build_reason_codes(
            prob_positive=p,
            positive_class_is_good=positive_class_is_good,
            penalty_cfg=penalty_cfg,
            descriptors=desc,
        )
        # risk_probability: P(unfavorable) for reporting — if y=1 is "good", risk ~ P(y=0).
        risk_p = (1.0 - p) if positive_class_is_good else p
        rows.append(
            {
                "candidate_id": row.get("candidate_id"),
                "smiles": row.get("smiles"),
                "label": int(row["label"]),
                "risk_probability": round(risk_p, 6),
                "predicted_prob_positive": round(p, 6),
                "prediction_confidence": round(confidence, 6),
                "priority_score": round(priority, 4),
                "descriptor_penalty_total": round(pen, 4),
                "reason_codes": "|".join(reasons),
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["priority_score", "prediction_confidence", "candidate_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def run_ranking_pipeline(
    model_config_path: str = "configs/model.yaml",
    ranking_config_path: str = "configs/ranking.yaml",
) -> dict[str, Any]:
    """End-to-end ranking: load config, model, features CSV, write ranked output."""
    r_cfg = load_yaml_config(ranking_config_path)["ranking"]
    model_key = str(r_cfg["model_key"])
    kind, bundle = load_inference_bundle(model_config_path, model_key)

    features_path = Path(r_cfg["default_features_path"])
    if not features_path.exists():
        msg = f"Features file not found: {features_path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(features_path)

    prob = predict_prob_positive_unified(kind, bundle, df)
    pen = r_cfg["penalties"]
    penalty_cfg = PenaltyConfig(
        high_logp_threshold=float(pen["high_logp_threshold"]),
        high_logp_penalty=float(pen["high_logp_penalty"]),
        high_mw_threshold=float(pen["high_mw_threshold"]),
        high_mw_penalty=float(pen["high_mw_penalty"]),
        high_tpsa_threshold=float(pen["high_tpsa_threshold"]),
        high_tpsa_penalty=float(pen["high_tpsa_penalty"]),
    )
    positive_is_good = bool(r_cfg["positive_class_is_good"])
    ranked = rank_candidates_frame(
        df,
        prob,
        positive_class_is_good=positive_is_good,
        penalty_cfg=penalty_cfg,
    )

    out_path = Path(r_cfg["ranked_output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)

    summary = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "model_key": model_key,
        "features_path": str(features_path),
        "rows": len(ranked),
        "positive_class_is_good": positive_is_good,
        "ranked_output_path": str(out_path),
    }
    summary_path = Path(r_cfg["ranking_summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    print(run_ranking_pipeline())
