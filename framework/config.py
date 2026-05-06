"""Configuration helpers for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

from pathlib import Path

from framework.hashing import compute_signature, file_sha256
from schema.experiment import ExperimentConfig
from schema.feature import FeatureExtractionConfigDump
from const.enum import Split


def load_config(path: str | Path) -> ExperimentConfig:
    if isinstance(path, str):
        path = Path(path)

    assert path.exists(), f"config path {path} does not exist"
    assert path.is_file(), "config path must point to a file"

    json_config = path.read_text()
    return ExperimentConfig.model_validate_json(json_config)


def get_split_extraction_config(config: ExperimentConfig, split: Split) -> FeatureExtractionConfigDump:
    payload = config.feature_extraction.model_dump()
    ids_path = config.feature_extraction.get_split_ids_path(split)

    return FeatureExtractionConfigDump(**payload, ids_path=ids_path, ids_sha256=file_sha256(ids_path))


def run_context_payload(config: ExperimentConfig) -> dict:
    return {
        "resolved_config_signature": compute_signature(config.model_dump(mode="json")),
        "training_feature_signature": get_split_extraction_config(config, Split.TRAINING).signature,
        "validation_feature_signature": get_split_extraction_config(config, Split.VALIDATION).signature,
        "test_feature_signature": get_split_extraction_config(config, Split.TEST).signature,
    }
