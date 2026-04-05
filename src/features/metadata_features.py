"""Metadata/context features for the locked MVP multimodal setup."""

from __future__ import annotations

import pandas as pd


def build_metadata_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic metadata features from ingestion outputs."""
    work = pd.DataFrame(index=df.index)
    work["meta_smiles_length"] = df["smiles"].astype(str).str.len().astype(int)
    work["meta_source_deepchem_bbbp"] = (df["data_source"] == "deepchem_bbbp:BBBP").astype(int)
    work["meta_candidate_id_numeric"] = (
        pd.to_numeric(df["candidate_id"], errors="coerce").fillna(-1).astype(float)
    )
    return work
