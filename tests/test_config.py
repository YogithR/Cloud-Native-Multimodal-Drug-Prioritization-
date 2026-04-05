from src.utils.config import load_yaml_config


def test_load_yaml_config_reads_base_config() -> None:
    cfg = load_yaml_config("configs/base.yaml")
    assert cfg["project"]["name"] == "drug-prioritization-platform"
