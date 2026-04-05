"""Run Phase 3 preprocessing and multimodal feature engineering."""

from src.features.build_feature_table import run_feature_pipeline


def main() -> None:
    counts = run_feature_pipeline("configs/data.yaml")
    print(f"Feature artifacts created. Split sizes: {counts}")


if __name__ == "__main__":
    main()
