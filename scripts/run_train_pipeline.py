"""Run Phase 4 baseline training pipeline."""

from src.models.train_baseline import run_training


def main() -> None:
    result = run_training("configs/model.yaml")
    print(f"Baseline training finished: {result}")


if __name__ == "__main__":
    main()

