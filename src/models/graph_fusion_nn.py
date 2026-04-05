"""Graph branch (GCN) + tabular MLP fusion for BBBP-style binary classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GraphFusionHyperParams:
    node_in_dim: int
    tab_dim: int
    gcn_hidden: int = 64
    gcn_layers: int = 3
    tab_hidden: int = 128
    fusion_dim: int = 64
    dropout: float = 0.15


class GraphFusionClassifier(nn.Module):
    """
    Two-branch encoder: GCN over atom graphs + MLP over tabular features, then fusion MLP.
    """

    def __init__(self, hp: GraphFusionHyperParams) -> None:
        super().__init__()
        self.hp = hp
        self.dropout = nn.Dropout(hp.dropout)
        self.gcn_weights = nn.ParameterList()
        dims = [hp.node_in_dim] + [hp.gcn_hidden] * hp.gcn_layers
        for i in range(len(dims) - 1):
            self.gcn_weights.append(nn.Parameter(torch.empty(dims[i], dims[i + 1])))
        for w in self.gcn_weights:
            nn.init.xavier_uniform_(w)

        self.tab_mlp = nn.Sequential(
            nn.Linear(hp.tab_dim, hp.tab_hidden),
            nn.ReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.tab_hidden, hp.fusion_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hp.gcn_hidden + hp.fusion_dim, hp.fusion_dim),
            nn.ReLU(),
            nn.Dropout(hp.dropout),
            nn.Linear(hp.fusion_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
        tab: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: [B, N, node_in], adj: [B, N, N], mask: [B, N], tab: [B, tab_dim]
        Returns logits [B].
        """
        h = x
        for layer_idx, w in enumerate(self.gcn_weights):
            support = torch.bmm(adj, h)
            h = torch.matmul(support, w)
            h = torch.relu(h)
            if layer_idx < len(self.gcn_weights) - 1:
                h = self.dropout(h)

        m = mask.unsqueeze(-1)
        h_sum = (h * m).sum(dim=1)
        denom = m.sum(dim=1).clamp(min=1e-6)
        g = h_sum / denom

        t = self.tab_mlp(tab)
        z = torch.cat([g, t], dim=-1)
        return self.head(z).squeeze(-1)

    @staticmethod
    def from_hyperparams_dict(d: dict[str, Any]) -> GraphFusionClassifier:
        hp = GraphFusionHyperParams(
            node_in_dim=int(d["node_in_dim"]),
            tab_dim=int(d["tab_dim"]),
            gcn_hidden=int(d["gcn_hidden"]),
            gcn_layers=int(d["gcn_layers"]),
            tab_hidden=int(d["tab_hidden"]),
            fusion_dim=int(d["fusion_dim"]),
            dropout=float(d["dropout"]),
        )
        return GraphFusionClassifier(hp)
