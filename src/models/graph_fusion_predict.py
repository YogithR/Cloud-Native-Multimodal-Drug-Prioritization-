"""Load graph-fusion artifacts and run batched inference on feature frames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from src.features.mol_graph import batch_graphs_from_smiles
from src.models.graph_fusion_nn import GraphFusionClassifier


def load_graph_fusion_artifact(path: str | Path) -> dict[str, Any]:
    """Load joblib bundle produced by train_graph_fusion."""
    return joblib.load(path)


def build_model_from_artifact(artifact: dict[str, Any]) -> GraphFusionClassifier:
    model = GraphFusionClassifier.from_hyperparams_dict(artifact["hyperparams"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def build_scaler_from_artifact(artifact: dict[str, Any]) -> Any:
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.mean_ = np.asarray(artifact["scaler_mean"], dtype=np.float64)
    sc.scale_ = np.asarray(artifact["scaler_scale"], dtype=np.float64)
    sc.var_ = sc.scale_**2
    sc.n_features_in_ = int(sc.mean_.shape[0])
    sc.feature_names_in_ = None
    return sc


@torch.no_grad()
def predict_proba_positive_batch(
    model: GraphFusionClassifier,
    artifact: dict[str, Any],
    df: pd.DataFrame,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Return P(y=1) as float array aligned with df rows."""
    dev = device or torch.device("cpu")
    model = model.to(dev)
    feature_columns: list[str] = artifact["feature_columns"]
    max_atoms = int(artifact["hyperparams"]["max_atoms"])
    scaler = build_scaler_from_artifact(artifact)

    x_tab = scaler.transform(df[feature_columns].to_numpy(dtype=np.float64))
    smiles = df["smiles"].astype(str).tolist()
    probs: list[float] = []
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        chunk = smiles[start:end]
        x_t, adj_t, mask_t = batch_graphs_from_smiles(chunk, max_atoms=max_atoms, device=dev)
        tab_t = torch.tensor(x_tab[start:end], dtype=torch.float32, device=dev)
        logits = model(x_t, adj_t, mask_t, tab_t)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(p.tolist())
    return np.asarray(probs, dtype=np.float64)


def predict_prob_series(
    artifact: dict[str, Any],
    df: pd.DataFrame,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> pd.Series:
    """P(y=1) as pandas Series indexed like df."""
    model = build_model_from_artifact(artifact)
    arr = predict_proba_positive_batch(model, artifact, df, device=device, batch_size=batch_size)
    return pd.Series(arr, index=df.index)
