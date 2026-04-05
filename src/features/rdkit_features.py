"""Deterministic molecular feature extraction using RDKit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def is_valid_smiles(smiles: str) -> bool:
    """Return True when SMILES can be parsed by RDKit."""
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


def _fingerprint_to_dict(mol: Any, radius: int, n_bits: int) -> dict[str, int]:
    generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    bits = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, bits)
    return {f"fp_{i:03d}": int(v) for i, v in enumerate(bits)}


def build_rdkit_features(
    smiles_series: pd.Series,
    fingerprint_radius: int = 2,
    fingerprint_bits: int = 128,
) -> pd.DataFrame:
    """Build deterministic RDKit descriptor + fingerprint features."""
    rows: list[dict[str, float | int]] = []
    for smiles in smiles_series.astype(str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            msg = f"Invalid SMILES encountered during feature generation: {smiles}"
            raise ValueError(msg)

        descriptor_row: dict[str, float | int] = {
            "desc_mol_wt": float(Descriptors.MolWt(mol)),
            "desc_logp": float(Descriptors.MolLogP(mol)),
            "desc_tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "desc_h_donors": int(Lipinski.NumHDonors(mol)),
            "desc_h_acceptors": int(Lipinski.NumHAcceptors(mol)),
            "desc_ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
            "desc_heavy_atoms": int(mol.GetNumHeavyAtoms()),
        }
        descriptor_row.update(_fingerprint_to_dict(mol, fingerprint_radius, fingerprint_bits))
        rows.append(descriptor_row)

    return pd.DataFrame(rows)
