"""Reason codes for ranking transparency."""

from __future__ import annotations

from typing import Any

from src.ranking.score import PenaltyConfig, compute_descriptor_penalties


def build_reason_codes(
    prob_positive: float,
    positive_class_is_good: bool,
    penalty_cfg: PenaltyConfig,
    descriptors: dict[str, Any] | None = None,
) -> list[str]:
    """Return short human-readable reason tags."""
    descriptors = descriptors or {}
    codes: list[str] = []
    if positive_class_is_good:
        if 0.4 < prob_positive < 0.6:
            codes.append("uncertain_prediction")
        if prob_positive >= 0.7:
            codes.append("high_predicted_permeability_confidence")
        elif prob_positive <= 0.3:
            codes.append("low_predicted_permeability_confidence")
    else:
        if 0.4 < prob_positive < 0.6:
            codes.append("uncertain_prediction")
        if prob_positive <= 0.3:
            codes.append("low_predicted_risk_confidence")
        elif prob_positive >= 0.7:
            codes.append("high_predicted_risk_confidence")

    logp = descriptors.get("desc_logp")
    mw = descriptors.get("desc_mol_wt")
    tpsa = descriptors.get("desc_tpsa")
    if logp is not None and float(logp) > penalty_cfg.high_logp_threshold:
        codes.append("descriptor_penalty_high_logp")
    if mw is not None and float(mw) > penalty_cfg.high_mw_threshold:
        codes.append("descriptor_penalty_high_mw")
    if tpsa is not None and float(tpsa) > penalty_cfg.high_tpsa_threshold:
        codes.append("descriptor_penalty_high_tpsa")

    total_pen = compute_descriptor_penalties(
        desc_logp=descriptors.get("desc_logp"),
        desc_mol_wt=descriptors.get("desc_mol_wt"),
        desc_tpsa=descriptors.get("desc_tpsa"),
        cfg=penalty_cfg,
    )
    if total_pen > 0:
        codes.append("descriptor_penalties_applied")
    return codes
