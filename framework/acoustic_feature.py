"""Feature extraction utilities for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import json
import pickle
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from streaming.base import MDSWriter

from framework.config import get_split_extraction_config
from framework.metadata import DOMAIN_TO_INDEX, SPECIES_TO_INDEX, load_id_list, parse_file_id
from framework.dataset import get_loader, split_feature_dir, load_split_metadata, training_split_stats_path
from schema.experiment import ExperimentConfig
from schema.data import SplitMetadata, SplitStatistics, ExtractedFeature
from const.filename import METADATA_FILENAME
from const.enum import Split


class LogMelSpectrogram(nn.Module):
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
        self.hop_length = hop_length
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


def load_waveform(path: str | Path, sample_rate: int, normalize_waveform: bool) -> np.ndarray:
    waveform, _ = librosa.load(Path(path), sr=sample_rate, mono=True)
    if normalize_waveform and waveform.size:
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
    return waveform.astype(np.float32)


def extract_log_mel_feature(
    audio_path: str | Path,
    extractor: LogMelSpectrogram,
    sample_rate: int,
    normalize_waveform: bool,
    device: torch.device,
) -> np.ndarray:
    waveform = load_waveform(audio_path, sample_rate, normalize_waveform)
    waveform_tensor = torch.tensor(waveform, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        feature = extractor(waveform_tensor)[0].detach().cpu().numpy().astype(np.float32)
    return feature


def split_feature_path(feature_root: str | Path, split_name: str) -> Path:
    return Path(feature_root) / f"{split_name.lower()}_features.pkl"


def feature_stats_path(feature_root: str | Path) -> Path:
    return Path(feature_root) / "training_feature_stats.json"


def extract_split_features(
    config: ExperimentConfig,
    split: Split,
    extractor: LogMelSpectrogram,
    device: torch.device,
) -> Path:
    feature_root = config.feature_root
    ids_path = config.feature_extraction.get_split_ids_path(split)
    split_ids = load_id_list(ids_path)

    total_items = len(split_ids)
    output_path = split_feature_dir(feature_root, split)

    output_path.mkdir(parents=True, exist_ok=True)

    with MDSWriter(out=str(output_path), columns=ExtractedFeature._MDS_COLUMNS) as sink:
        for file_id in split_ids:
            species, domain = parse_file_id(file_id)
            audio_path = config.feature_extraction.dataset_root / f"{file_id}.wav"
            feature = extract_log_mel_feature(
                audio_path,
                extractor,
                config.feature_extraction.sample_rate,
                config.feature_extraction.normalize_waveform,
                device,
            )
            sample = ExtractedFeature(
                feature=feature,
                file_id=file_id,
                num_frames=int(feature.shape[0]),
                feature_dim=int(feature.shape[1]),
                species=species,
                species_label=SPECIES_TO_INDEX[species],
                domain=domain,
                domain_label=DOMAIN_TO_INDEX[domain],
                audio_path=audio_path,
            )
            sink.write(sample.model_dump())

    metadata = SplitMetadata(
        split=split,
        num_items=total_items,
        config=get_split_extraction_config(config, split),
    )

    with open(output_path / METADATA_FILENAME, "w+") as f:
        f.write(metadata.model_dump_json(indent=2))

    return output_path


def compute_training_split_stats(feature_root: Path) -> SplitStatistics:
    split = Split.TRAINING
    metadata = load_split_metadata(feature_root, split)
    loader = get_loader(
        feature_root,
        split=split,
        batch_size=1,
        num_workers=0,
        normalize_features=False,
        collate_fn=lambda x: x[0],
    )

    feature_sum = np.zeros((metadata.config.n_mels,), dtype=np.float64)
    feature_sq_sum = np.zeros((metadata.config.n_mels,), dtype=np.float64)
    total_frames: int = 0

    for item in loader:
        item: ExtractedFeature
        feature = item.feature.astype(np.float64)

        feature_sum += feature.sum(axis=0, dtype=np.float64)
        feature_sq_sum += np.square(feature, dtype=np.float64).sum(axis=0, dtype=np.float64)
        total_frames += feature.shape[0]

    mean = feature_sum / total_frames
    variance = np.maximum(feature_sq_sum / total_frames - np.square(mean), 1e-12)
    std = np.sqrt(variance)

    return SplitStatistics(
        num_frames=total_frames,
        config=metadata.config,
        mean=mean.astype(np.float32).tolist(),
        std=std.astype(np.float32).tolist(),
    )


def save_split_stats(stats: SplitStatistics, feature_root: str | Path) -> Path:
    output_path = training_split_stats_path(feature_root)
    with open(output_path, "w+", encoding="utf-8") as f:
        f.write(stats.model_dump_json(indent=2))

    return output_path
