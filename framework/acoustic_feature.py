"""Feature extraction utilities for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Union
from abc import ABC, abstractmethod

import librosa
import numpy as np
import torch
import torch.nn as nn
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
import torchaudio

from framework.config import config_signature, feature_signature_payload
from framework.metadata import DOMAIN_TO_INDEX, SPECIES_TO_INDEX, load_id_list, parse_file_id


class FeatureExtractor(ABC, nn.Module):
    @abstractmethod
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        pass

class LogSpectrogram(FeatureExtractor):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
    ) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
       
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype="amplitude")

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrogram = self.spectrogram_extractor(waveforms)
        return self.db_transform(spectrogram.abs()).squeeze(1)

class LogMelSpectrogram(FeatureExtractor):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        n_mels: int,
        fmin: int,
        fmax: int,
    ) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrogram = self.spectrogram_extractor(waveforms)
        logmel = self.logmel_extractor(spectrogram)
        return logmel.squeeze(1)


def load_waveform(path: Union[str, Path], sample_rate: int, normalize_waveform: bool) -> np.ndarray:
    waveform, _ = librosa.load(Path(path), sr=sample_rate, mono=True)
    if normalize_waveform and waveform.size:
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
    return waveform.astype(np.float32)


def extract_feature(
    audio_path: Union[str, Path],
    extractor: FeatureExtractor,
    sample_rate: int,
    normalize_waveform: bool,
    device: torch.device,
) -> np.ndarray:
    waveform = load_waveform(audio_path, sample_rate, normalize_waveform)
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        feature = extractor(waveform_tensor)[0].detach().cpu().numpy().astype(np.float32)
    return feature


def split_feature_path(feature_root: Union[str, Path], split_name: str) -> Path:
    return Path(feature_root) / f"{split_name.lower()}_features.pkl"


def feature_stats_path(feature_root: Union[str, Path]) -> Path:
    return Path(feature_root) / "training_feature_stats.json"


def extract_split_features(
    config: Dict,
    split_name: str,
    extractor: FeatureExtractor,
    device: torch.device,
) -> Path:
    feature_root = Path(config["feature_root"])
    feature_root.mkdir(parents=True, exist_ok=True)
    records = []
    ids_path = config[{"training": "train_ids_path", "validation": "val_ids_path", "test": "test_ids_path"}[split_name]]
    file_ids = load_id_list(ids_path)
    total_items = len(file_ids)

    for index, file_id in enumerate(file_ids, start=1):
        species, domain = parse_file_id(file_id)
        audio_path = Path(config["dataset_root"]) / f"{file_id}.wav"
        feature = extract_feature(
            audio_path,
            extractor,
            config["feature_extraction"]["sample_rate"],
            config["feature_extraction"]["normalize_waveform"],
            device,
        )
        print(
            f"[{split_name}] {index}/{total_items} | "
            f"id={file_id} | feature_shape={tuple(feature.shape)}"
        )
        records.append(
            {
                "file_id": file_id,
                "feature": feature,
                "num_frames": int(feature.shape[0]),
                "feature_dim": int(feature.shape[1]),
                "species": species,
                "species_label": SPECIES_TO_INDEX[species],
                "domain": domain,
                "domain_label": DOMAIN_TO_INDEX[domain],
                "audio_path": str(audio_path),
            }
        )

    payload = {
        "split": split_name,
        "num_items": len(records),
        "config_signature": config_signature(feature_signature_payload(config, split_name)),
        "config_payload": feature_signature_payload(config, split_name),
        "items": records,
    }
    output_path = split_feature_path(feature_root, split_name)
    with open(output_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def compute_training_feature_stats(feature_pickle_path: Union[str, Path]) -> Dict:
    with open(feature_pickle_path, "rb") as handle:
        payload = pickle.load(handle)

    feature_sum = None
    feature_sq_sum = None
    total_frames = 0

    for item in payload["items"]:
        feature = item["feature"]
        if feature_sum is None:
            feature_sum = feature.sum(axis=0, dtype=np.float64)
            feature_sq_sum = np.square(feature, dtype=np.float64).sum(axis=0, dtype=np.float64)
        else:
            feature_sum += feature.sum(axis=0, dtype=np.float64)
            feature_sq_sum += np.square(feature, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_frames += feature.shape[0]

    mean = feature_sum / total_frames
    variance = np.maximum(feature_sq_sum / total_frames - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    return {
        "num_frames": int(total_frames),
        "feature_config_signature": payload["config_signature"],
        "feature_config_payload": payload["config_payload"],
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
    }


def save_feature_stats(stats: Dict, feature_root: Union[str, Path]) -> Path:
    output_path = feature_stats_path(feature_root)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return output_path
