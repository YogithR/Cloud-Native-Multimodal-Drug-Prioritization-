"""Generate three-way comparison report (baseline vs RF vs graph fusion)."""

from __future__ import annotations

import json

from src.models.compare_models import run_three_way_comparison


def main() -> None:
    report = run_three_way_comparison()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
