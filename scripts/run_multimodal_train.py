"""Run Phase 5 multimodal training pipeline."""

from src.models.train_multimodal import run_multimodal_training


def main() -> None:
    result = run_multimodal_training("configs/model.yaml")
    print("Multimodal training finished.")
    print(result)


if __name__ == "__main__":
    main()
