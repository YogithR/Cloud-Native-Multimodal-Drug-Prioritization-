import pandas as pd

from src.ingestion.pubchem_enrich import add_source_metadata
from src.ingestion.validate_schema import run_dataset_sanity_checks, validate_and_standardize_admet


def test_validate_schema_standardizes_columns() -> None:
    df = pd.DataFrame({"Drug": ["CCO", "CCC"], "Y": [1, 0]})
    out = validate_and_standardize_admet(df)
    assert list(out.columns[:3]) == ["candidate_id", "smiles", "label"]
    assert out["label"].tolist() == [1, 0]


def test_add_source_metadata_adds_data_source_column() -> None:
    df = pd.DataFrame({"candidate_id": ["c1"], "smiles": ["CCO"], "label": [1]})
    out = add_source_metadata(df, source_name="TDC:BBBP")
    assert out["data_source"].iloc[0] == "TDC:BBBP"


def test_run_dataset_sanity_checks_returns_summary() -> None:
    df = pd.DataFrame({"candidate_id": ["c1", "c2"], "smiles": ["CCO", "CCC"], "label": [1, 0]})
    summary = run_dataset_sanity_checks(df)
    assert summary["rows"] == 2
    assert summary["positive_label_rate"] == 0.5
