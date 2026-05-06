"""Dataset helpers for reading precomputed feature files.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import random
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming.base import StreamingDataset

from framework.config import get_split_extraction_config
from schema.data import (
    SplitMetadata,
    SplitStatistics,
    ExtractedFeature,
    DataLoaderBatch,
)
from schema.experiment import ExperimentConfig
from const.filename import (
    TRAINING_SPLIT_STATS_FILENAME,
    METADATA_FILENAME,
)
from const.enum import Split


def split_feature_dir(feature_root: str | Path, split: Split) -> Path:
    return Path(feature_root) / f"{split.value}"


def training_split_stats_path(feature_root: str | Path) -> Path:
    return Path(feature_root) / TRAINING_SPLIT_STATS_FILENAME


def load_split_metadata(feature_root: Path, split: Split) -> SplitMetadata:
    split_dir = split_feature_dir(feature_root, split)
    with open(split_dir / METADATA_FILENAME, "r") as f:
        return SplitMetadata.model_validate_json(f.read())


def load_training_split_stats(feature_root: Path) -> SplitStatistics:
    with open(training_split_stats_path(feature_root), "r") as f:
        return SplitStatistics.model_validate_json(f.read())


def validate_split_metadata(metadata: SplitMetadata, config: ExperimentConfig, split: Split) -> None:
    if metadata.config.signature != get_split_extraction_config(config, split).signature:
        raise ValueError("Extraction config of the provided split does not match the current configuration")


def validate_training_split_stats(stats: SplitStatistics, config: ExperimentConfig) -> None:
    if stats.config.signature != get_split_extraction_config(config, Split.TRAINING).signature:
        raise ValueError(
            "Extraction config of the provided split statistics does not match the current configuration."
        )


def pad_collate_fn(batch: list[ExtractedFeature]) -> DataLoaderBatch:
    lengths = torch.tensor([item.feature.shape[0] for item in batch], dtype=torch.long)
    padded = pad_sequence(
        [torch.from_numpy(item.feature) for item in batch],
        batch_first=True,
        padding_value=0,
    ).type(torch.float32)

    return DataLoaderBatch(
        file_id=[item.file_id for item in batch],
        features=padded,
        lengths=lengths,
        species_labels=torch.tensor([item.species_label for item in batch], dtype=torch.long),
        domain_labels=torch.tensor([item.domain_label for item in batch], dtype=torch.long),
        species=[item.species for item in batch],
        domain=[item.domain for item in batch],
        audio_path=[item.audio_path for item in batch],
    )


def get_loader[T](
    feature_root: Path,
    split: Split,
    batch_size: int = 64,
    num_workers: int = 4,
    max_train_frames: int | None = None,
    training: bool = False,
    shuffle: bool = False,
    pin_memory: bool = False,
    config: ExperimentConfig | None = None,
    normalize_features: bool = False,
    verify_config_signature: bool = False,
    verify_stats_signature: bool = False,
    min_prefetch_samples: int = 1000,
    collate_fn: Callable[[list[ExtractedFeature]], T] = pad_collate_fn,
) -> DataLoader[T]:
    metadata = load_split_metadata(feature_root, split)

    if verify_config_signature:
        assert config is not None
        validate_split_metadata(metadata, config, split)

    transforms_list = []

    if normalize_features:
        stats = load_training_split_stats(feature_root)
        if verify_stats_signature:
            assert config is not None
            validate_training_split_stats(stats, config)

        transforms_list.append(
            Normalize(
                np.array(stats.mean, dtype=np.float32),
                np.array(stats.std, dtype=np.float32),
            )
        )

    if training and max_train_frames is not None:
        transforms_list.append(MaybeCrop(max_train_frames))

    transform = transforms.Compose(transforms_list)

    def _collate_fn(raw_batch: list[dict]) -> T:
        batch = [ExtractedFeature.model_validate(item) for item in raw_batch]
        for item in batch:
            item.feature = transform(item.feature.copy())

        return collate_fn(batch)

    split_path = split_feature_dir(feature_root, split)
    return DataLoader(
        StreamingDataset(
            local=str(split_path),
            batch_size=batch_size,
            predownload=max(min_prefetch_samples, batch_size * 3),
            shuffle=shuffle,
        ),
        collate_fn=_collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


class Normalize:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, feature: np.ndarray) -> np.ndarray:
        return (feature - self.mean) / np.maximum(self.std, 1e-8)


class MaybeCrop:
    def __init__(self, max_train_frames: int) -> None:
        self.max_train_frames = max_train_frames

    def __call__(self, feature: np.ndarray) -> np.ndarray:
        if feature.shape[0] <= self.max_train_frames:
            return feature
        start = random.randint(0, feature.shape[0] - self.max_train_frames)
        return feature[start : start + self.max_train_frames]
