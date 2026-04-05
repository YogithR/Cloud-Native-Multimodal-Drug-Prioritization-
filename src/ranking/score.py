"""Priority scoring from model risk/goodness probability and descriptor penalties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PenaltyConfig:
    high_logp_threshold: float
    high_logp_penalty: float
    high_mw_threshold: float
    high_mw_penalty: float
    high_tpsa_threshold: float
    high_tpsa_penalty: float


def compute_descriptor_penalties(
    desc_logp: float | None,
    desc_mol_wt: float | None,
    desc_tpsa: float | None,
    cfg: PenaltyConfig,
) -> float:
    """Sum simple rule-based penalties (deterministic, interpretable)."""
    penalty = 0.0
    if desc_logp is not None and desc_logp > cfg.high_logp_threshold:
        penalty += cfg.high_logp_penalty
    if desc_mol_wt is not None and desc_mol_wt > cfg.high_mw_threshold:
        penalty += cfg.high_mw_penalty
    if desc_tpsa is not None and desc_tpsa > cfg.high_tpsa_threshold:
        penalty += cfg.high_tpsa_penalty
    return penalty


def compute_base_priority_score(
    prob_positive: float,
    positive_class_is_good: bool,
) -> float:
    """Map P(y=1) to 0..100 base score before penalties."""
    if positive_class_is_good:
        return 100.0 * prob_positive
    return 100.0 * (1.0 - prob_positive)


def compute_priority_score(
    prob_positive: float,
    positive_class_is_good: bool,
    penalty_cfg: PenaltyConfig,
    descriptors: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """
    Return (priority_score, total_penalty).

    priority_score = base_score - descriptor_penalties, clamped to [0, 100].
    """
    base = compute_base_priority_score(prob_positive, positive_class_is_good)
    descriptors = descriptors or {}
    pen = compute_descriptor_penalties(
        desc_logp=descriptors.get("desc_logp"),
        desc_mol_wt=descriptors.get("desc_mol_wt"),
        desc_tpsa=descriptors.get("desc_tpsa"),
        cfg=penalty_cfg,
    )
    score = max(0.0, min(100.0, base - pen))
    return score, pen
