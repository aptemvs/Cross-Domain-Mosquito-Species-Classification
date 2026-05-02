"""Shared utility functions for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import csv
import json
import logging
import math
import os
import random
import socket
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from schema.experiment_config import ExperimentConfig
from schema.trial_config import TrialConfig
from const.model_backend import ModelBackend


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_output_dir(output_root: str | Path, experiment_name: str) -> Path:
    output_dir = Path(output_root) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def acquire_experiment_lock(output_dir: Path, experiment_name: str) -> Path | None:
    lock_path = output_dir / ".experiment.lock"
    payload = {
        "experiment_name": experiment_name,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        file_descriptor = os.open(str(lock_path), flags)
    except FileExistsError:
        return None
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return lock_path


def release_experiment_lock(lock_path: Path | None) -> None:
    if lock_path is not None and lock_path.exists():
        lock_path.unlink()


def make_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def append_metrics(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_json(path: Path, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: Path) -> dict | list:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_std(values: list[float]) -> tuple[float, float]:
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return mean_value, math.sqrt(variance)


def format_mean_std(values: list[float]) -> str:
    mean_value, std_value = mean_std(values)
    return f"{mean_value:.6f} +- {std_value:.6f}"


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_table(path: Path, report_rows: list[dict]) -> None:
    headers = ["metric", "validation", "test"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in report_rows:
        lines.append(f"| {row['metric']} | {row['validation']} | {row['test']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device, collate_fn) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=True
    )


def split_feature_path(config: ExperimentConfig, split_name: str) -> Path:
    return config.feature_root / f"{split_name.lower()}_features.pkl"


def training_stats_path(config: ExperimentConfig) -> Path:
    return config.feature_root / "training_feature_stats.json"


def max_train_frames(config: TrialConfig) -> int:
    return max(
        1,
        int(
            round(
                config.train_crop_seconds
                * config.feature_extraction.sample_rate
                / config.feature_extraction.hop_length
            )
        ),
    )


def build_model(config: TrialConfig, device: torch.device):
    backend = config.backend

    match backend.model:
        case ModelBackend.MTRCNN:
            from framework.model_baseline import MTRCNNClassifier

            return MTRCNNClassifier(
                config=config,
                num_species_classes=len(SPECIES_NAMES),
                num_domain_classes=len(DOMAIN_NAMES),
            ).to(device)
        case ModelBackend.EFFICIENTAT:
            from framework.model_efficientat import EfficientATClassifier

            return EfficientATClassifier(
                config=config,
                num_species_classes=len(SPECIES_NAMES),
                num_domain_classes=len(DOMAIN_NAMES),
            ).to(device)
        case _:
            raise ValueError(f"Unknown model backend {backend!r}")
