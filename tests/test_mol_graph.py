"""Molecular graph construction tests."""

import numpy as np

from src.features.mol_graph import (
    batch_graphs_from_smiles,
    node_feature_dim,
    normalize_adjacency,
    smiles_to_graph,
)


def test_smiles_to_graph_shapes() -> None:
    x, adj, mask = smiles_to_graph("CCO", max_atoms=32)
    f = node_feature_dim()
    assert x.shape == (32, f)
    assert adj.shape == (32, 32)
    assert mask.sum() == 3  # C, C, O


def test_normalize_adjacency_symmetric() -> None:
    adj = np.eye(4, dtype=np.float32)
    adj[0, 1] = adj[1, 0] = 1.0
    n = normalize_adjacency(adj)
    assert np.allclose(n, n.T)


def test_batch_graphs_from_smiles_runs(torch_cpu) -> None:
    x, adj, m = batch_graphs_from_smiles(
        ["CCO", "c1ccccc1"], max_atoms=16, device=torch_cpu.device("cpu")
    )
    assert x.shape[0] == 2
    assert adj.shape[0] == 2
    assert m.shape[0] == 2
