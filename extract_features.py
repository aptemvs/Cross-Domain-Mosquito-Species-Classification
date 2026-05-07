"""Feature extraction entry point for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse

import torch

from framework.acoustic_feature import (
    LogMelSpectrogram,
    compute_training_split_stats,
    extract_split_features,
    save_split_stats,
)
from framework.config import get_split_extraction_config, load_config
from framework.dataset import (
    load_training_split_stats,
    training_split_stats_path,
    split_feature_dir,
    load_split_metadata,
)
from framework.utilization import choose_device
from schema.feature import FeatureExtractionConfig
from const.enum import Split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract log-mel features for all splits.")
    parser.add_argument("--config", type=str, default="configs/default_experiment.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_feature_extractor(config: FeatureExtractionConfig, device: torch.device) -> LogMelSpectrogram:
    extractor = LogMelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.f_min,
        fmax=config.f_max,
    ).to(device)
    extractor.eval()
    return extractor


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(config.device)
    extractor = build_feature_extractor(config.feature_extraction, device)
    feature_root = config.feature_root

    for split in list(Split):
        output_dir = split_feature_dir(feature_root, split)

        if output_dir.exists() and not args.overwrite:
            try:
                expected_signature = get_split_extraction_config(config, split).signature
                metadata = load_split_metadata(feature_root, split)
                if metadata.config.signature == expected_signature:
                    print(f"loading from {output_dir}")
                    continue
            except Exception:
                pass

        print(f"extracting {split.value} features to {output_dir}")
        extract_split_features(
            config=config,
            split=split,
            extractor=extractor,
            device=device,
        )

    stats_path = training_split_stats_path(feature_root)
    if stats_path.exists() and not args.overwrite:
        try:
            training_signature = get_split_extraction_config(config, Split.TRAINING).signature
            stats = load_training_split_stats(feature_root)
            if stats.config.signature == training_signature:
                print(f"loading from {stats_path}")
                return
        except Exception:
            pass

    print(f"computing training feature stats to {stats_path}")
    stats = compute_training_split_stats(feature_root)
    save_split_stats(stats, feature_root)


if __name__ == "__main__":
    main()
