"""Feature extraction entry point for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
from pathlib import Path

import torch

from framework.acoustic_feature import (
    LogMelSpectrogram,
    compute_training_feature_stats,
    extract_split_features,
    feature_stats_path,
    save_feature_stats,
    split_feature_path,
)
from framework.config import config_signature, feature_signature_payload, load_config
from framework.dataset import load_feature_payload
from framework.utilization import choose_device, load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract log-mel features for all splits.")
    parser.add_argument("--config", type=str, default="configs/default_experiment.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_feature_extractor(config: dict, device: torch.device) -> LogMelSpectrogram:
    extractor = LogMelSpectrogram(
        sample_rate=config["sample_rate"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        win_length=config["win_length"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    ).to(device)
    extractor.eval()
    return extractor


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(config["device"])
    extractor = build_feature_extractor(config["feature_extraction"], device)
    feature_root = Path(config["feature_root"])

    split_jobs = [
        "training",
        "validation",
        "test",
    ]
    for split_name in split_jobs:
        output_path = split_feature_path(feature_root, split_name)
        expected_signature = config_signature(feature_signature_payload(config, split_name))
        if output_path.exists() and not args.overwrite:
            payload = load_feature_payload(output_path)
            if payload.get("config_signature") == expected_signature:
                print(f"loading from {output_path}")
                continue
        print(f"extracting {split_name} features to {output_path}")
        extract_split_features(
            config=config,
            split_name=split_name,
            extractor=extractor,
            device=device,
        )

    stats_path = feature_stats_path(feature_root)
    training_signature = config_signature(feature_signature_payload(config, "training"))
    if stats_path.exists() and not args.overwrite:
        stats_payload = load_json(stats_path)
        if stats_payload.get("feature_config_signature") == training_signature:
            print(f"loading from {stats_path}")
            return

    print(f"computing training feature stats to {stats_path}")
    stats = compute_training_feature_stats(split_feature_path(feature_root, "training"))
    save_feature_stats(stats, feature_root)


if __name__ == "__main__":
    main()
