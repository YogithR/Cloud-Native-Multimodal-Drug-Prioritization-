"""Ranking endpoints for candidate prioritization."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.deps import rank_candidates
from src.api.schemas import BatchRankRequest, CandidateInput, RankResponse

router = APIRouter(tags=["rank"])


@router.post("/rank", response_model=RankResponse)
def rank_one(candidate: CandidateInput) -> RankResponse:
    ranked = rank_candidates([candidate.model_dump()])
    return RankResponse(**ranked.iloc[0].to_dict())


@router.post("/batch-rank", response_model=list[RankResponse])
def rank_batch(payload: BatchRankRequest) -> list[RankResponse]:
    ranked = rank_candidates([c.model_dump() for c in payload.candidates])
    return [RankResponse(**row) for row in ranked.to_dict(orient="records")]
