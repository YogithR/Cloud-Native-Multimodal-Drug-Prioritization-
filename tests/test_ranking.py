"""Tests for Phase 6 ranking and scoring."""

import pandas as pd

from src.ranking.explain import build_reason_codes
from src.ranking.ranker import rank_candidates_frame
from src.ranking.score import PenaltyConfig, compute_base_priority_score, compute_priority_score


def test_compute_base_priority_score_positive_is_good() -> None:
    assert compute_base_priority_score(0.8, positive_class_is_good=True) == 80.0


def test_compute_base_priority_score_risk_framing() -> None:
    assert compute_base_priority_score(0.2, positive_class_is_good=False) == 80.0


def test_compute_priority_score_applies_penalties() -> None:
    cfg = PenaltyConfig(
        high_logp_threshold=3.0,
        high_logp_penalty=10.0,
        high_mw_threshold=500.0,
        high_mw_penalty=0.0,
        high_tpsa_threshold=140.0,
        high_tpsa_penalty=0.0,
    )
    score, pen = compute_priority_score(
        0.5,
        positive_class_is_good=True,
        penalty_cfg=cfg,
        descriptors={"desc_logp": 5.0},
    )
    assert pen == 10.0
    assert score == 40.0


def test_rank_candidates_frame_orders_by_priority() -> None:
    df = pd.DataFrame(
        {
            "candidate_id": [1, 2],
            "smiles": ["CCO", "CCC"],
            "label": [1, 0],
            "desc_logp": [1.0, 1.0],
            "desc_mol_wt": [46.0, 44.0],
            "desc_tpsa": [20.0, 20.0],
        }
    )
    prob = pd.Series([0.9, 0.1])
    cfg = PenaltyConfig(
        high_logp_threshold=5.0,
        high_logp_penalty=5.0,
        high_mw_threshold=500.0,
        high_mw_penalty=5.0,
        high_tpsa_threshold=140.0,
        high_tpsa_penalty=3.0,
    )
    out = rank_candidates_frame(df, prob, positive_class_is_good=True, penalty_cfg=cfg)
    assert out.iloc[0]["candidate_id"] == 1
    assert out.iloc[0]["rank"] == 1
    assert "prediction_confidence" in out.columns


def test_build_reason_codes_contains_high_confidence() -> None:
    cfg = PenaltyConfig(
        high_logp_threshold=5.0,
        high_logp_penalty=5.0,
        high_mw_threshold=500.0,
        high_mw_penalty=5.0,
        high_tpsa_threshold=140.0,
        high_tpsa_penalty=3.0,
    )
    codes = build_reason_codes(0.8, True, cfg, {})
    assert "high_predicted_permeability_confidence" in codes
