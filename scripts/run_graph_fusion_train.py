"""Train graph-enhanced multimodal fusion model (GCN + tabular MLP)."""

from __future__ import annotations

import json

from src.models.train_graph_fusion import run_graph_fusion_training


def main() -> None:
    summary = run_graph_fusion_training()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
