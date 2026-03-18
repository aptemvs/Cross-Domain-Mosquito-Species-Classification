"""Prediction entry point for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from framework.acoustic_feature import LogMelSpectrogram, extract_log_mel_feature
from framework.config import config_signature, feature_signature_payload, load_config
from framework.dataset import load_feature_stats, validate_feature_stats_payload
from framework.metadata import SPECIES_NAMES
from framework.utilization import build_model, choose_device, training_stats_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a single mosquito audio file.")
    parser.add_argument("--config", type=str, default="configs/default_experiment.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
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


def normalize_feature(feature: np.ndarray, stats_path: Path, normalize_features: bool) -> np.ndarray:
    if not normalize_features:
        return feature
    mean, std = load_feature_stats(stats_path)
    return (feature - mean) / np.maximum(std, 1e-6)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(config["device"])

    extractor = build_feature_extractor(config, device)
    feature = extract_log_mel_feature(
        audio_path=Path(args.audio),
        extractor=extractor,
        sample_rate=config["sample_rate"],
        normalize_waveform=config["normalize_waveform"],
        device=device,
    )
    print(f"loading from {args.checkpoint}")
    expected_training_stats_signature = config_signature(feature_signature_payload(config, "training"))
    validate_feature_stats_payload(training_stats_path(config), expected_training_stats_signature)
    feature = normalize_feature(feature, training_stats_path(config), config["normalize_features"])
    if config["normalize_features"]:
        print(f"loading from {training_stats_path(config)}")

    features = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
    lengths = torch.tensor([feature.shape[0]], dtype=torch.long, device=device)

    model = build_model(config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(features, lengths)
        probs = torch.softmax(outputs["species_logits"], dim=1)[0]
        pred_index = int(torch.argmax(probs).item())
        prob_values = probs.detach().cpu().tolist()

    result = {
        "predicted_species": SPECIES_NAMES[pred_index],
        "probabilities": {label: round(prob, 6) for label, prob in zip(SPECIES_NAMES, prob_values)},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
