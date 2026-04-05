"""Run Phase 2 ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.download_tdc import fetch_public_admet_csv, save_dataframe
from src.ingestion.pubchem_enrich import add_source_metadata
from src.ingestion.validate_schema import run_dataset_sanity_checks, validate_and_standardize_admet
from src.utils.config import load_yaml_config


def run() -> None:
    cfg = load_yaml_config("configs/data.yaml")["data"]
    dataset_name = cfg["dataset_name"]
    dataset_url = cfg["dataset_url"]

    raw_dir = Path(cfg["raw_dir"])
    interim_dir = Path(cfg["interim_dir"])
    processed_dir = Path(cfg["processed_dir"])
    raw_file = raw_dir / f"{dataset_name.lower()}_raw.csv"
    interim_file = interim_dir / f"{dataset_name.lower()}_validated.csv"
    processed_file = processed_dir / f"{dataset_name.lower()}_processed.csv"

    raw_df = fetch_public_admet_csv(dataset_url=dataset_url)
    save_dataframe(raw_df, raw_file)

    validated_df = validate_and_standardize_admet(
        raw_df,
        smiles_col=cfg["smiles_column"],
        label_col=cfg["label_column"],
        candidate_id_col=cfg["candidate_id_column"],
    )
    sanity = run_dataset_sanity_checks(validated_df)
    save_dataframe(validated_df, interim_file)

    enriched_df = add_source_metadata(
        validated_df,
        source_name=f"{cfg['dataset_source']}:{dataset_name}",
    )
    save_dataframe(enriched_df, processed_file)

    print(f"Saved raw data: {raw_file}")
    print(f"Saved validated data: {interim_file}")
    print(f"Saved processed data: {processed_file}")
    print(f"Rows in processed dataset: {len(enriched_df)}")
    print(f"Sanity summary: {sanity}")


if __name__ == "__main__":
    run()
