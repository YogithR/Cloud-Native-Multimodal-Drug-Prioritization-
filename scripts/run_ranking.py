"""Run Phase 6 candidate ranking pipeline (no API)."""

from src.ranking.ranker import run_ranking_pipeline


def main() -> None:
    summary = run_ranking_pipeline(
        model_config_path="configs/model.yaml",
        ranking_config_path="configs/ranking.yaml",
    )
    print("Ranking finished.")
    print(summary)


if __name__ == "__main__":
    main()
