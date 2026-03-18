"""Configuration helpers for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Union


FEATURE_CONFIG_KEYS = [
    "dataset_root",
    "sample_rate",
    "normalize_waveform",
    "n_fft",
    "hop_length",
    "win_length",
    "n_mels",
    "fmin",
    "fmax",
]


def load_config(path: Union[str, Path]) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    return resolve_config(config)


def resolve_config(config: Dict) -> Dict:
    config = dict(config)
    sample_rate = config["sample_rate"]
    config["n_mels"] = 64
    config["fmin"] = 0
    config["fmax"] = int(sample_rate / 2)
    config["win_length"] = max(1, int(math.floor(1024 * (sample_rate / 16000))))
    config["hop_length"] = max(1, int(math.floor(160 * (sample_rate / 16000))))
    config["n_fft"] = config["win_length"]
    return config


def config_subset(config: Dict, keys: List[str]) -> Dict:
    return {key: config[key] for key in keys}


def config_signature(payload: Dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def feature_signature_payload(config: Dict, split_name: str) -> Dict:
    split_to_ids_key = {
        "training": "train_ids_path",
        "validation": "val_ids_path",
        "test": "test_ids_path",
    }
    payload = config_subset(config, FEATURE_CONFIG_KEYS)
    ids_path = config[split_to_ids_key[split_name]]
    payload["split"] = split_name
    payload["ids_path"] = ids_path
    payload["ids_sha256"] = file_sha256(ids_path)
    return payload


def run_context_payload(config: Dict) -> Dict:
    return {
        "resolved_config_signature": config_signature(config),
        "training_feature_signature": config_signature(feature_signature_payload(config, "training")),
        "validation_feature_signature": config_signature(feature_signature_payload(config, "validation")),
        "test_feature_signature": config_signature(feature_signature_payload(config, "test")),
    }
