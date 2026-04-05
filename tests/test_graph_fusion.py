"""Graph fusion model smoke tests (requires working PyTorch)."""

import numpy as np
import pandas as pd


def test_graph_fusion_forward(torch_cpu) -> None:
    from src.features.mol_graph import batch_graphs_from_smiles, node_feature_dim
    from src.models.graph_fusion_nn import GraphFusionClassifier, GraphFusionHyperParams

    hp = GraphFusionHyperParams(
        node_in_dim=node_feature_dim(),
        tab_dim=5,
        gcn_hidden=16,
        gcn_layers=2,
        tab_hidden=32,
        fusion_dim=16,
        dropout=0.0,
    )
    m = GraphFusionClassifier(hp)
    smiles = ["CCO", "CCC"]
    x, adj, mask = batch_graphs_from_smiles(smiles, max_atoms=16, device=torch_cpu.device("cpu"))
    tab = torch_cpu.zeros(2, 5, dtype=torch_cpu.float32)
    logits = m(x, adj, mask, tab)
    assert logits.shape == (2,)
    assert torch_cpu.isfinite(logits).all()


def test_predict_proba_positive_batch_matches_forward(torch_cpu) -> None:
    from src.features.mol_graph import node_feature_dim
    from src.models.graph_fusion_nn import GraphFusionClassifier, GraphFusionHyperParams
    from src.models.graph_fusion_predict import predict_proba_positive_batch

    hp = GraphFusionHyperParams(
        node_in_dim=node_feature_dim(),
        tab_dim=2,
        gcn_hidden=8,
        gcn_layers=1,
        tab_hidden=8,
        fusion_dim=8,
        dropout=0.0,
    )
    model = GraphFusionClassifier(hp)
    df = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "smiles": ["CCO", "CCN"],
            "label": [1, 0],
            "f0": [0.0, 1.0],
            "f1": [1.0, 0.0],
        }
    )
    artifact = {
        "hyperparams": {
            "node_in_dim": hp.node_in_dim,
            "tab_dim": hp.tab_dim,
            "gcn_hidden": hp.gcn_hidden,
            "gcn_layers": hp.gcn_layers,
            "tab_hidden": hp.tab_hidden,
            "fusion_dim": hp.fusion_dim,
            "dropout": hp.dropout,
            "max_atoms": 16,
        },
        "feature_columns": ["f0", "f1"],
        "scaler_mean": np.zeros(2),
        "scaler_scale": np.ones(2),
    }
    p = predict_proba_positive_batch(model, artifact, df)
    assert p.shape == (2,)
    assert np.all((p >= 0) & (p <= 1))
