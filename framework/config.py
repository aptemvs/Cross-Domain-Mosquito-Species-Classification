"""Configuration helpers for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

from pathlib import Path

from framework.utilization import compute_signature, file_sha256


def load_config(path: str | Path) -> ExperimentConfig:
    if isinstance(path, str):
        path = Path(path)

    assert path.exists(), f"config path {path} does not exist"
    assert path.is_file(), "config path must point to a file"

    json_config = path.read_text()
    return ExperimentConfig.model_validate_json(json_config)


def config_subset(config: dict, keys: list[str]) -> dict:
    return {key: config[key] for key in keys}


def feature_signature_payload(config: ExperimentConfig, split_name: str) -> dict:
    split_to_ids_key = {
        "training": "train_ids_path",
        "validation": "val_ids_path",
        "test": "test_ids_path",
    }
    payload = config.feature_extraction.model_dump(mode="json")

    # Exclude paths to files
    include_keys = set(payload.keys()).difference(
        ["dataset_root"], split_to_ids_key.values()
    )
    payload = config_subset(payload, list(include_keys))

    ids_path = getattr(config.feature_extraction, split_to_ids_key[split_name])
    payload["split"] = split_name
    payload["ids_sha256"] = file_sha256(ids_path)
    return payload


def run_context_payload(config: ExperimentConfig) -> dict:
    return {
        "resolved_config_signature": config_signature(config.model_dump(mode="json")),
        "training_feature_signature": config_signature(
            feature_signature_payload(config, "training")
        ),
        "validation_feature_signature": config_signature(
            feature_signature_payload(config, "validation")
        ),
        "test_feature_signature": config_signature(
            feature_signature_payload(config, "test")
        ),
    }
