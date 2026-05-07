"""Evaluation entry point for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch

from framework.config import load_config
from framework.dataset import get_loader
from framework.engine import balanced_accuracy, evaluate_model
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import build_model, choose_device, generate_trials
from schema.trial import TrialConfig
from schema.experiment import ExperimentConfig
from const.enum import Split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved mosquito classifier.")
    parser.add_argument("--config", type=str, default="configs/default_experiment.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=Split, choices=[Split.VALIDATION, Split.TEST], default=Split.TEST)
    parser.add_argument("--metrics-out", type=str, default=None)
    parser.add_argument("--predictions-out", type=str, default=None)
    return parser.parse_args()


def save_prediction_rows(path: str | Path, rows) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def split_summary_path(config: ExperimentConfig) -> Path:
    return config.feature_extraction.train_ids_path.parent / "split_summary.json"


def load_unseen_domain_by_species(config: ExperimentConfig) -> dict[str, str]:
    path = split_summary_path(config)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("unseen_domain_by_species", {})


def annotate_evaluation_partition(prediction_rows: list[dict], unseen_domain_by_species: dict[str, str]) -> list[dict]:
    annotated_rows = []
    for row in prediction_rows:
        annotated_row = dict(row)
        true_species_label = annotated_row.get("true_species_label")
        true_domain_label = annotated_row.get("true_domain_label")
        partition = None
        if true_species_label in unseen_domain_by_species and true_domain_label is not None:
            partition = "unseen" if unseen_domain_by_species[true_species_label] == true_domain_label else "seen"
        annotated_row["evaluation_partition"] = partition
        annotated_rows.append(annotated_row)
    return annotated_rows


def subset_balanced_accuracy(prediction_rows: list[dict]) -> float | None:
    if not prediction_rows:
        return None
    preds = torch.tensor([row["predicted_species_index"] for row in prediction_rows], dtype=torch.long)
    labels = torch.tensor([row["true_species_index"] for row in prediction_rows], dtype=torch.long)
    return balanced_accuracy(preds, labels, len(SPECIES_NAMES))


def append_official_metrics(metrics: dict, prediction_rows: list[dict], unseen_domain_by_species: dict[str, str]) -> dict:
    annotated_rows = annotate_evaluation_partition(prediction_rows, unseen_domain_by_species)
    seen_rows = [row for row in annotated_rows if row["evaluation_partition"] == "seen"]
    unseen_rows = [row for row in annotated_rows if row["evaluation_partition"] == "unseen"]

    ba_seen = subset_balanced_accuracy(seen_rows)
    ba_unseen = subset_balanced_accuracy(unseen_rows)

    metrics["num_seen_samples"] = len(seen_rows)
    metrics["num_unseen_samples"] = len(unseen_rows)
    metrics["BA_seen"] = ba_seen
    metrics["BA_unseen"] = ba_unseen
    metrics["DSG"] = abs(ba_unseen - ba_seen) if ba_seen is not None and ba_unseen is not None else None
    return {
        "metrics": metrics,
        "predictions": annotated_rows,
    }


def evaluate_checkpoint(config: TrialConfig, checkpoint_path: str | Path, split: Split, return_predictions: bool = True) -> dict:
    config = deepcopy(config)
    device = choose_device(config.device)

    dataloader = get_loader(
        feature_root=config.feature_root,
        split=split,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        max_train_frames=None,
        training=False,
        shuffle=False,
        pin_memory=device.type == "cuda",
        config=config,
        normalize_features=True,
        verify_config_signature=True,
        verify_stats_signature=True,
    )

    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    result = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        num_species_classes=len(SPECIES_NAMES),
        num_domain_classes=len(DOMAIN_NAMES),
        species_names=SPECIES_NAMES,
        domain_names=DOMAIN_NAMES,
        return_predictions=return_predictions,
    )
    if return_predictions:
        metrics = result["metrics"]
        predictions = result["predictions"]
        if split == Split.TEST:
            official_result = append_official_metrics(metrics, predictions, load_unseen_domain_by_species(config))
            metrics = official_result["metrics"]
            predictions = official_result["predictions"]
    else:
        metrics = result
        predictions = None
    metrics["split"] = split.value
    metrics["checkpoint_path"] = str(checkpoint_path)
    if return_predictions:
        return {
            "metrics": metrics,
            "predictions": predictions,
        }
    return {"metrics": metrics}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    trials = list(generate_trials(config))
    if len(trials) > 1:
        raise ValueError(
            "Config describes more than one trial. Use run_multiple_experiments.py"
        )

    result = evaluate_checkpoint(trials[0], args.checkpoint, args.split, return_predictions=True)
    metrics = result["metrics"]
    predictions = result["predictions"]
    if args.metrics_out:
        with open(args.metrics_out, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
    if args.predictions_out:
        save_prediction_rows(args.predictions_out, predictions)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
