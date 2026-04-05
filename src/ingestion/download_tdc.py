"""Download public ADMET-style datasets for Phase 2."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


def fetch_public_admet_csv(dataset_url: str) -> pd.DataFrame:
    """Fetch an ADMET dataset CSV from a public URL."""
    response = requests.get(dataset_url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    if df.empty:
        msg = f"Dataset URL returned no rows: {dataset_url}"
        raise ValueError(msg)
    return df


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist DataFrame as CSV and return saved path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
