"""Evaluate graph fusion model on val/test and log metrics."""

from __future__ import annotations

import json

from src.models.evaluate import run_graph_fusion_evaluation


def main() -> None:
    report = run_graph_fusion_evaluation()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
