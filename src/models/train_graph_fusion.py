"""Train graph + tabular fusion neural model (Phase 12b advanced upgrade)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.features.mol_graph import batch_graphs_from_smiles, node_feature_dim
from src.models.graph_fusion_nn import GraphFusionClassifier, GraphFusionHyperParams
from src.models.train_baseline import get_feature_columns
from src.utils.config import load_yaml_config
from src.utils.mlflow_tracking import (
    configure_experiment,
    is_mlflow_enabled,
    log_file_artifact,
    log_params_flat,
)


def run_graph_fusion_training(config_path: str = "configs/model.yaml") -> dict[str, Any]:
    cfg = load_yaml_config(config_path)["model"]
    gf = cfg["graph_fusion"]
    random_state = int(cfg["random_state"])
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    train_df = pd.read_csv(gf["train_features_path"])
    val_df = pd.read_csv(gf["val_features_path"])
    feature_columns = get_feature_columns(train_df)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns].to_numpy(dtype=np.float64))

    max_atoms = int(gf["max_atoms"])
    node_in = node_feature_dim()
    tab_dim = len(feature_columns)
    hp = GraphFusionHyperParams(
        node_in_dim=node_in,
        tab_dim=tab_dim,
        gcn_hidden=int(gf["gcn_hidden"]),
        gcn_layers=int(gf["gcn_layers"]),
        tab_hidden=int(gf["tab_hidden"]),
        fusion_dim=int(gf["fusion_dim"]),
        dropout=float(gf["dropout"]),
    )
    model = GraphFusionClassifier(hp)
    device = torch.device("cpu")
    model.to(device)

    pos = float((train_df["label"] == 1).sum())
    neg = float((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(gf["learning_rate"]))

    batch_size = int(gf["batch_size"])

    best_val_auc = -1.0
    best_state: dict[str, Any] | None = None
    epochs = int(gf["epochs"])
    patience = int(gf["early_stopping_patience"])
    bad_epochs = 0

    hyperparams_dict: dict[str, Any] = {
        "node_in_dim": node_in,
        "tab_dim": tab_dim,
        "gcn_hidden": hp.gcn_hidden,
        "gcn_layers": hp.gcn_layers,
        "tab_hidden": hp.tab_hidden,
        "fusion_dim": hp.fusion_dim,
        "dropout": hp.dropout,
        "max_atoms": max_atoms,
    }

    epochs_ran = 0
    for epoch in range(epochs):
        epochs_ran = epoch + 1
        model.train()
        # Deterministic shuffling per epoch; avoids default DataLoader collate on pandas rows.
        rng = np.random.default_rng(random_state + epoch)
        indices = rng.permutation(len(train_df))
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            df_chunk = train_df.iloc[batch_idx]
            labels = torch.tensor(
                df_chunk["label"].astype(np.float32).to_numpy(),
                dtype=torch.float32,
                device=device,
            )
            smiles = df_chunk["smiles"].astype(str).tolist()
            x_t, adj_t, mask_t = batch_graphs_from_smiles(
                smiles,
                max_atoms=max_atoms,
                device=device,
            )
            x_tab = scaler.transform(df_chunk[feature_columns].to_numpy(dtype=np.float64))
            tab_t = torch.tensor(x_tab, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            logits = model(x_t, adj_t, mask_t, tab_t)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs = _eval_probs(model, val_df, feature_columns, scaler, max_atoms, device)
        val_y = val_df["label"].astype(int).to_numpy()
        val_auc = float(roc_auc_score(val_y, val_probs))

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model_path = Path(gf["model_artifact_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact: dict[str, Any] = {
        "model_type": "graph_fusion_nn",
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "feature_columns": feature_columns,
        "scaler_mean": scaler.mean_.astype(np.float64),
        "scaler_scale": scaler.scale_.astype(np.float64),
        "hyperparams": hyperparams_dict,
        "threshold": float(gf["threshold"]),
        "train_config": {
            "epochs_ran": epochs_ran,
            "best_val_roc_auc": best_val_auc,
            "random_state": random_state,
        },
    }
    joblib.dump(artifact, model_path)

    train_summary_path = Path(gf["train_summary_path"])
    train_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_timestamp_utc": datetime.now(UTC).isoformat(),
        "model_type": "GraphFusionClassifier",
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "feature_count": len(feature_columns),
        "graph_branch": "GCN",
        "tabular_branch": "MLP",
        "fusion": "concat + MLP head",
        "best_val_roc_auc": best_val_auc,
        "epochs_ran": epochs_ran,
        "hyperparams": hyperparams_dict,
        "model_artifact_path": str(model_path),
    }
    with train_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    if is_mlflow_enabled():
        configure_experiment()
        with mlflow.start_run(run_name="graph_fusion_train"):
            mlflow.set_tags(
                {
                    "model_family": "graph_fusion",
                    "graph_encoder": "GCN",
                    "fusion": "concat_mlp",
                }
            )
            log_params_flat(
                {
                    **{k: str(v) for k, v in hyperparams_dict.items()},
                    "train_rows": len(train_df),
                    "feature_count": len(feature_columns),
                    "epochs_ran": epochs_ran,
                }
            )
            mlflow.log_metric("best_val_roc_auc", float(best_val_auc))
            log_file_artifact(str(model_path))
            log_file_artifact(str(train_summary_path))

    return summary


@torch.no_grad()
def _eval_probs(
    model: GraphFusionClassifier,
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
    max_atoms: int,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    probs: list[float] = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start : start + batch_size]
        smiles = chunk["smiles"].astype(str).tolist()
        x_t, adj_t, mask_t = batch_graphs_from_smiles(smiles, max_atoms=max_atoms, device=device)
        x_tab = scaler.transform(chunk[feature_columns].to_numpy(dtype=np.float64))
        tab_t = torch.tensor(x_tab, dtype=torch.float32, device=device)
        logits = model(x_t, adj_t, mask_t, tab_t)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.extend(p.tolist())
    return np.asarray(probs, dtype=np.float64)


if __name__ == "__main__":
    out = run_graph_fusion_training()
    print(json.dumps(out, indent=2))
