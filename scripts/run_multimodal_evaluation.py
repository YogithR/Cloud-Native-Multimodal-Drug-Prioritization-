"""Run Phase 5 multimodal evaluation and comparison pipeline."""

from src.models.evaluate import build_comparison_report, run_evaluation


def main() -> None:
    report = run_evaluation("configs/model.yaml", model_key="multimodal")
    comparison = build_comparison_report("configs/model.yaml")
    print("Multimodal evaluation finished.")
    print(report)
    print("Baseline vs multimodal comparison:")
    print(comparison)


if __name__ == "__main__":
    main()
