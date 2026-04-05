"""SMILES → padded molecular graph tensors for neural encoders (RDKit, deterministic)."""

from __future__ import annotations

from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem


def _hybrid_one_hot(hybrid: rdchem.HybridizationType) -> list[float]:
    order = (
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
        rdchem.HybridizationType.UNSPECIFIED,
    )
    return [1.0 if hybrid == h else 0.0 for h in order]


def atom_features(atom: rdchem.Atom) -> list[float]:
    """Fixed-length handcrafted atom features (interpretable, no learned embeddings)."""
    sym = atom.GetSymbol()
    element_bits = [0.0] * 8
    try:
        idx = ("C", "N", "O", "F", "S", "Cl", "Br").index(sym)
    except ValueError:
        idx = 7
    element_bits[idx] = 1.0

    feats: list[float] = [
        float(atom.GetAtomicNum()) / 100.0,
        float(atom.GetTotalDegree()) / 6.0,
        float(atom.GetFormalCharge()) / 4.0,
        float(atom.GetTotalNumHs(includeNeighbors=True)) / 6.0,
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        float(atom.GetMass()) / 200.0,
    ]
    feats.extend(_hybrid_one_hot(atom.GetHybridization()))
    feats.extend(element_bits)
    return feats


def smiles_to_graph(
    smiles: str,
    max_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (node_features, adjacency, mask) with shapes
    [max_atoms, F], [max_atoms, max_atoms], [max_atoms].
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atoms = list(mol.GetAtoms())
    n_raw = len(atoms)
    if n_raw > max_atoms:
        atoms = atoms[:max_atoms]

    f_dim = len(atom_features(atoms[0]))
    x = np.zeros((max_atoms, f_dim), dtype=np.float32)
    adj = np.eye(max_atoms, dtype=np.float32)
    mask = np.zeros((max_atoms,), dtype=np.float32)

    idx_map: dict[int, int] = {}
    for new_i, atom in enumerate(atoms):
        old_idx = atom.GetIdx()
        idx_map[old_idx] = new_i
        x[new_i] = np.asarray(atom_features(atom), dtype=np.float32)
        mask[new_i] = 1.0

    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in idx_map and b in idx_map:
            i, j = idx_map[a], idx_map[b]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    return x, adj, mask


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalization D^{-1/2} A D^{-1/2} (self-loops already in adj)."""
    d = adj.sum(axis=-1)
    d_inv_sqrt = np.zeros_like(d, dtype=np.float64)
    np.divide(1.0, np.sqrt(np.maximum(d, 1e-12)), out=d_inv_sqrt, where=d > 0)
    return (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)


def batch_graphs_from_smiles(
    smiles_list: list[str],
    max_atoms: int,
    device: Any,
) -> tuple[Any, Any, Any]:
    """Stack graphs into batched tensors on device (lazy-imports torch)."""
    import torch

    xs: list[np.ndarray] = []
    adjs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for smi in smiles_list:
        x, adj, mask = smiles_to_graph(smi, max_atoms=max_atoms)
        adjs.append(normalize_adjacency(adj))
        xs.append(x)
        masks.append(mask)
    x_t = torch.tensor(np.stack(xs, axis=0), dtype=torch.float32, device=device)
    adj_t = torch.tensor(np.stack(adjs, axis=0), dtype=torch.float32, device=device)
    mask_t = torch.tensor(np.stack(masks, axis=0), dtype=torch.float32, device=device)
    return x_t, adj_t, mask_t


def node_feature_dim() -> int:
    """Dimension of atom_features() — used to size the graph encoder."""
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    return len(atom_features(mol.GetAtomWithIdx(0)))
