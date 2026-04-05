"""API dependencies for model loading and feature construction."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import HTTPException

from src.features.metadata_features import build_metadata_features
from src.features.rdkit_features import build_rdkit_features, is_valid_smiles
from src.ranking.inference import load_inference_bundle, predict_prob_positive_unified
from src.ranking.ranker import rank_candidates_frame
from src.ranking.score import PenaltyConfig
from src.utils.config import load_yaml_config


@lru_cache(maxsize=1)
def get_runtime_config() -> dict[str, Any]:
    data_cfg = load_yaml_config("configs/data.yaml")["data"]
    model_cfg = load_yaml_config("configs/model.yaml")["model"]
    ranking_cfg = load_yaml_config("configs/ranking.yaml")["ranking"]
    return {"data": data_cfg, "model": model_cfg, "ranking": ranking_cfg}


@lru_cache(maxsize=1)
def get_inference_bundle() -> tuple[str, dict[str, Any]]:
    cfg = get_runtime_config()
    model_key = str(cfg["ranking"]["model_key"])
    return load_inference_bundle("configs/model.yaml", model_key)


def _build_model_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    cfg = get_runtime_config()
    fp_cfg = cfg["data"]["features"]
    raw = pd.DataFrame(candidates)
    if raw.empty:
        raise HTTPException(status_code=400, detail="At least one candidate is required.")
    if (~raw["smiles"].map(is_valid_smiles)).any():
        raise HTTPException(status_code=400, detail="One or more SMILES strings are invalid.")

    meta_df = build_metadata_features(raw)
    rdkit_df = build_rdkit_features(
        raw["smiles"],
        fingerprint_radius=int(fp_cfg["fingerprint_radius"]),
        fingerprint_bits=int(fp_cfg["fingerprint_bits"]),
    )
    frame = pd.concat(
        [raw[["candidate_id", "smiles"]].reset_index(drop=True), meta_df, rdkit_df],
        axis=1,
    )
    return frame


def predict_candidates(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    kind, bundle = get_inference_bundle()
    cfg = get_runtime_config()
    positive_is_good = bool(cfg["ranking"]["positive_class_is_good"])
    frame = _build_model_frame(candidates)
    prob = predict_prob_positive_unified(kind, bundle, frame)
    conf = (prob - 0.5).abs() * 2.0

    out = frame[["candidate_id", "smiles"]].copy()
    out["predicted_prob_positive"] = prob.round(6)
    out["prediction_confidence"] = conf.round(6)
    out["risk_probability"] = ((1.0 - prob) if positive_is_good else prob).round(6)
    return out.reset_index(drop=True)


def rank_candidates(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    kind, bundle = get_inference_bundle()
    cfg = get_runtime_config()
    ranking_cfg = cfg["ranking"]
    frame = _build_model_frame(candidates)
    prob = predict_prob_positive_unified(kind, bundle, frame)
    penalties = ranking_cfg["penalties"]
    penalty_cfg = PenaltyConfig(
        high_logp_threshold=float(penalties["high_logp_threshold"]),
        high_logp_penalty=float(penalties["high_logp_penalty"]),
        high_mw_threshold=float(penalties["high_mw_threshold"]),
        high_mw_penalty=float(penalties["high_mw_penalty"]),
        high_tpsa_threshold=float(penalties["high_tpsa_threshold"]),
        high_tpsa_penalty=float(penalties["high_tpsa_penalty"]),
    )
    frame_for_rank = frame.copy()
    frame_for_rank["label"] = 0
    ranked = rank_candidates_frame(
        frame_for_rank,
        prob,
        positive_class_is_good=bool(ranking_cfg["positive_class_is_good"]),
        penalty_cfg=penalty_cfg,
    )
    return ranked.drop(columns=["label"])
