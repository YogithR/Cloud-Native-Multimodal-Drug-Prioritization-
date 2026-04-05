"""API schemas used by endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str = "0.1.0"


class CandidateInput(BaseModel):
    candidate_id: str
    smiles: str
    data_source: str = "deepchem_bbbp:BBBP"


class PredictResponse(BaseModel):
    candidate_id: str
    smiles: str
    predicted_prob_positive: float
    risk_probability: float
    prediction_confidence: float


class RankResponse(BaseModel):
    rank: int
    candidate_id: str
    smiles: str
    predicted_prob_positive: float
    risk_probability: float
    prediction_confidence: float
    priority_score: float
    descriptor_penalty_total: float
    reason_codes: str


class BatchRankRequest(BaseModel):
    candidates: list[CandidateInput]
