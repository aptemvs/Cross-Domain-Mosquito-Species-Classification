"""Configuration helpers for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import hashlib
import json
from pathlib import Path
from pydantic import TypeAdapter

from model.experiment_config import ExperimentConfig

def load_config(path: str | Path) -> ExperimentConfig:
    if isinstance(path, str):
        path = Path(path)

    assert path.exists(), f"config path {path} does not exist"
    assert path.is_file(), "config path must point to a file"

    json_config = path.read_text()
    return TypeAdapter(ExperimentConfig).validate_json(json_config)


def config_subset(config: dict, keys: list[str]) -> dict:
    return {key: config[key] for key in keys}


def config_signature(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def feature_signature_payload(config: ExperimentConfig, split_name: str) -> dict:
    split_to_ids_key = {
        "training": "train_ids_path",
        "validation": "val_ids_path",
        "test": "test_ids_path",
    }
    payload = config.feature_extraction.model_dump()
    ids_path = payload[split_to_ids_key[split_name]]
    payload["split"] = split_name
    payload["ids_path"] = ids_path
    payload["ids_sha256"] = file_sha256(ids_path)
    return payload


def run_context_payload(config: ExperimentConfig) -> dict:
    return {
        "resolved_config_signature": config_signature(config.model_dump()),
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
