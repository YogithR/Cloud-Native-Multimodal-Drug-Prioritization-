"""Prediction endpoints for model inference."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import predict_candidates
from src.api.schemas import CandidateInput, PredictResponse

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictResponse)
def predict_one(candidate: CandidateInput) -> PredictResponse:
    out = predict_candidates([candidate.model_dump()]).iloc[0].to_dict()
    return PredictResponse(**out)
