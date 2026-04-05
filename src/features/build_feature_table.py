"""Preprocessing, splitting, and multimodal feature table creation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.metadata_features import build_metadata_features
from src.features.rdkit_features import build_rdkit_features, is_valid_smiles
from src.utils.config import load_yaml_config


def clean_dataset_for_modeling(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning rules for modeling."""
    work = df.copy()
    required = ["candidate_id", "smiles", "label", "data_source"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        msg = f"Missing required columns for preprocessing: {missing}"
        raise ValueError(msg)

    work = work.dropna(subset=["candidate_id", "smiles", "label"]).copy()
    work["smiles"] = work["smiles"].astype(str).str.strip()
    work = work[work["smiles"] != ""]
    work = work[work["smiles"].map(is_valid_smiles)]
    work["label"] = work["label"].astype(int)
    work = work[work["label"].isin([0, 1])]
    work = work.drop_duplicates(subset=["smiles", "label"]).reset_index(drop=True)
    return work


def create_reproducible_splits(
    df: pd.DataFrame,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> dict[str, pd.DataFrame]:
    """Split data into train/val/test with deterministic random state."""
    if round(train_size + val_size + test_size, 6) != 1.0:
        raise ValueError("Split sizes must sum to 1.0.")

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=df["label"],
    )
    val_ratio_within_temp = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_ratio_within_temp),
        random_state=seed,
        stratify=temp_df["label"],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def build_multimodal_feature_table(
    split_df: pd.DataFrame,
    fingerprint_radius: int,
    fingerprint_bits: int,
) -> pd.DataFrame:
    """Combine metadata and molecular features into one deterministic table."""
    rdkit_df = build_rdkit_features(
        split_df["smiles"],
        fingerprint_radius=fingerprint_radius,
        fingerprint_bits=fingerprint_bits,
    )
    meta_df = build_metadata_features(split_df)
    id_label_df = split_df[["candidate_id", "smiles", "label"]].reset_index(drop=True)
    features_df = pd.concat([id_label_df, meta_df.reset_index(drop=True), rdkit_df], axis=1)
    return features_df


def save_feature_artifacts(
    artifacts: dict[str, pd.DataFrame],
    processed_dir: Path,
    schema: dict[str, Any],
) -> None:
    """Persist feature splits and schema as reproducible artifacts."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in artifacts.items():
        split_df.to_csv(processed_dir / f"bbbp_{split_name}_features.csv", index=False)
    with (processed_dir / "bbbp_feature_schema.json").open("w", encoding="utf-8") as file:
        json.dump(schema, file, indent=2)


def run_feature_pipeline(config_path: str = "configs/data.yaml") -> dict[str, int]:
    """Run Phase 3 preprocessing + feature engineering pipeline."""
    cfg = load_yaml_config(config_path)["data"]
    seed = int(cfg["seed"])
    split_cfg = cfg["split"]
    feat_cfg = cfg["features"]

    input_path = Path(cfg["processed_dir"]) / f"{cfg['dataset_name'].lower()}_processed.csv"
    if not input_path.exists():
        msg = f"Input dataset not found for Phase 3: {input_path}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(input_path)
    cleaned = clean_dataset_for_modeling(df)
    splits = create_reproducible_splits(
        cleaned,
        seed=seed,
        train_size=float(split_cfg["train_size"]),
        val_size=float(split_cfg["val_size"]),
        test_size=float(split_cfg["test_size"]),
    )

    artifacts: dict[str, pd.DataFrame] = {}
    for split_name, split_df in splits.items():
        artifacts[split_name] = build_multimodal_feature_table(
            split_df,
            fingerprint_radius=int(feat_cfg["fingerprint_radius"]),
            fingerprint_bits=int(feat_cfg["fingerprint_bits"]),
        )

    sample = artifacts["train"]
    schema = {
        "id_columns": ["candidate_id", "smiles"],
        "target_column": "label",
        "metadata_feature_columns": [c for c in sample.columns if c.startswith("meta_")],
        "molecular_descriptor_columns": [c for c in sample.columns if c.startswith("desc_")],
        "molecular_fingerprint_columns": [c for c in sample.columns if c.startswith("fp_")],
        "fingerprint_bits": int(feat_cfg["fingerprint_bits"]),
        "fingerprint_radius": int(feat_cfg["fingerprint_radius"]),
        "split_seed": seed,
    }
    processed_dir = Path(cfg["processed_dir"])
    save_feature_artifacts(artifacts, processed_dir, schema)

    cleaned.to_csv(processed_dir / "bbbp_cleaned_dataset.csv", index=False)
    return {name: len(frame) for name, frame in artifacts.items()}
