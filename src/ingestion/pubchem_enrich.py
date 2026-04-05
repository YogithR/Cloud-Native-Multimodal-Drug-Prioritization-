"""Optional lightweight metadata enrichment stubs."""

from __future__ import annotations

import pandas as pd


def add_source_metadata(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add source metadata column for multimodal context."""
    work = df.copy()
    work["data_source"] = source_name
    return work
