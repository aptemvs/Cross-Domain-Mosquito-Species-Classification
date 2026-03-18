"""Dataset helpers for reading precomputed feature files.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def load_feature_payload(path: Union[str, Path]) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_feature_stats(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    return mean, std


def validate_feature_payload(payload: Dict, expected_signature: Optional[str]) -> None:
    if expected_signature is None:
        return
    if payload.get("config_signature") != expected_signature:
        raise ValueError("Feature file does not match the current configuration.")


def validate_feature_stats_payload(path: Union[str, Path], expected_signature: Optional[str]) -> None:
    if expected_signature is None:
        return
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("feature_config_signature") != expected_signature:
        raise ValueError("Feature statistics file does not match the current training feature configuration.")


class MosquitoFeatureDataset(Dataset):
    def __init__(
        self,
        feature_pickle_path: Union[str, Path],
        feature_stats_path: Optional[Union[str, Path]] = None,
        max_train_frames: Optional[int] = None,
        training: bool = False,
        normalize_features: bool = True,
        expected_feature_signature: Optional[str] = None,
        expected_stats_signature: Optional[str] = None,
    ) -> None:
        payload = load_feature_payload(feature_pickle_path)
        validate_feature_payload(payload, expected_feature_signature)
        self.samples = payload["items"]
        self.training = training
        self.max_train_frames = max_train_frames
        self.normalize_features = normalize_features and feature_stats_path is not None
        self.feature_mean = None
        self.feature_std = None
        if self.normalize_features:
            validate_feature_stats_payload(feature_stats_path, expected_stats_signature)
            self.feature_mean, self.feature_std = load_feature_stats(feature_stats_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_crop(self, feature: np.ndarray) -> np.ndarray:
        if not self.training or not self.max_train_frames or feature.shape[0] <= self.max_train_frames:
            return feature
        start = random.randint(0, feature.shape[0] - self.max_train_frames)
        return feature[start : start + self.max_train_frames]

    def _normalize(self, feature: np.ndarray) -> np.ndarray:
        if not self.normalize_features:
            return feature
        return (feature - self.feature_mean) / np.maximum(self.feature_std, 1e-8)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        feature = sample["feature"].astype(np.float32)
        feature = self._maybe_crop(feature)
        feature = self._normalize(feature)
        return {
            "file_id": sample["file_id"],
            "feature": torch.tensor(feature, dtype=torch.float32),
            "length": feature.shape[0],
            "species_label": sample["species_label"],
            "domain_label": sample["domain_label"],
            "species": sample["species"],
            "domain": sample["domain"],
            "audio_path": sample["audio_path"],
        }


def pad_collate_fn(batch: List[Dict]) -> Dict:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item()) if len(lengths) else 0
    feature_dim = int(batch[0]["feature"].shape[1]) if batch else 0
    padded = torch.zeros(len(batch), max_length, feature_dim, dtype=torch.float32)

    for idx, item in enumerate(batch):
        padded[idx, : item["length"], :] = item["feature"]

    return {
        "file_id": [item["file_id"] for item in batch],
        "features": padded,
        "lengths": lengths,
        "species_labels": torch.tensor([item["species_label"] for item in batch], dtype=torch.long),
        "domain_labels": torch.tensor([item["domain_label"] for item in batch], dtype=torch.long),
        "species": [item["species"] for item in batch],
        "domain": [item["domain"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
    }
