"""Schema validation and cleaning for ingestion outputs."""

from __future__ import annotations

import pandas as pd


def validate_and_standardize_admet(
    df: pd.DataFrame,
    smiles_col: str = "Drug",
    label_col: str = "Y",
    candidate_id_col: str = "candidate_id",
) -> pd.DataFrame:
    """Validate required columns and return standardized DataFrame."""
    missing = [c for c in (smiles_col, label_col) if c not in df.columns]
    if missing:
        msg = f"Missing required columns: {missing}"
        raise ValueError(msg)

    work = df.copy()
    work = work.rename(columns={smiles_col: "smiles", label_col: "label"})
    work["smiles"] = work["smiles"].astype(str).str.strip()
    work = work[work["smiles"] != ""]
    work = work.dropna(subset=["label"])
    work["label"] = work["label"].astype(int)

    if candidate_id_col in work.columns:
        work = work.rename(columns={candidate_id_col: "candidate_id"})
    elif "candidate_id" not in work.columns:
        work["candidate_id"] = [f"cand_{i+1}" for i in range(len(work))]

    # Keep core columns first for downstream consistency.
    core = ["candidate_id", "smiles", "label"]
    remaining = [c for c in work.columns if c not in core]
    return work[core + remaining].reset_index(drop=True)


def run_dataset_sanity_checks(df: pd.DataFrame) -> dict[str, float | int]:
    """Run lightweight sanity checks and return summary stats."""
    if df.empty:
        raise ValueError("Sanity check failed: dataset is empty.")

    null_smiles = int(df["smiles"].isna().sum())
    null_label = int(df["label"].isna().sum())
    if null_smiles > 0 or null_label > 0:
        msg = f"Sanity check failed: nulls found (smiles={null_smiles}, label={null_label})."
        raise ValueError(msg)

    unique_labels = set(df["label"].unique().tolist())
    if not unique_labels.issubset({0, 1}):
        msg = f"Sanity check failed: non-binary labels found: {sorted(unique_labels)}"
        raise ValueError(msg)

    duplicate_pairs = int(df.duplicated(subset=["smiles", "label"]).sum())
    label_rate = float(df["label"].mean())
    return {
        "rows": int(len(df)),
        "duplicate_smiles_label_pairs": duplicate_pairs,
        "positive_label_rate": round(label_rate, 4),
        "negative_label_rate": round(1 - label_rate, 4),
    }
