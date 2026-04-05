import pandas as pd

from src.features.build_feature_table import (
    build_multimodal_feature_table,
    clean_dataset_for_modeling,
    create_reproducible_splits,
)
from src.features.metadata_features import build_metadata_features
from src.features.rdkit_features import build_rdkit_features


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": [1, 2, 3, 4, 5, 6],
            "smiles": ["CCO", "CCC", "CCN", "CCCl", "CCBr", "CCF"],
            "label": [1, 0, 1, 0, 1, 0],
            "data_source": ["deepchem_bbbp:BBBP"] * 6,
        }
    )


def test_clean_dataset_for_modeling_filters_invalid_rows() -> None:
    df = _sample_df()
    df.loc[0, "smiles"] = ""
    cleaned = clean_dataset_for_modeling(df)
    assert len(cleaned) == 5
    assert cleaned["label"].isin([0, 1]).all()


def test_reproducible_splits_are_deterministic() -> None:
    df_extra = _sample_df().assign(candidate_id=lambda x: x["candidate_id"] + 100)
    df = pd.concat([_sample_df(), df_extra])
    split_a = create_reproducible_splits(df, seed=42, train_size=0.6, val_size=0.2, test_size=0.2)
    split_b = create_reproducible_splits(df, seed=42, train_size=0.6, val_size=0.2, test_size=0.2)
    assert split_a["train"]["candidate_id"].tolist() == split_b["train"]["candidate_id"].tolist()


def test_rdkit_and_metadata_feature_shapes() -> None:
    df = _sample_df().head(2)
    rdkit_df = build_rdkit_features(df["smiles"], fingerprint_radius=2, fingerprint_bits=16)
    meta_df = build_metadata_features(df)
    fused = build_multimodal_feature_table(df, fingerprint_radius=2, fingerprint_bits=16)
    assert rdkit_df.shape[1] == 7 + 16
    assert "meta_smiles_length" in meta_df.columns
    assert "desc_mol_wt" in fused.columns
