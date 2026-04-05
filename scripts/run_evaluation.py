"""Run Phase 4 baseline evaluation pipeline."""

from src.models.evaluate import run_evaluation


def main() -> None:
    report = run_evaluation("configs/model.yaml")
    print("Baseline evaluation finished.")
    print(report)


if __name__ == "__main__":
    main()
