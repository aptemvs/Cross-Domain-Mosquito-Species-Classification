"""Generate tables and figures for the technical report.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
import csv
import json
import math
import pickle
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from framework.config import load_config, run_context_payload
from framework.metadata import SPECIES_NAMES, parse_file_id
from framework.utilization import build_model


RUN_DIR_PATTERN = re.compile(r"^MTRCNN_seed(\d+)_B(\d+)_E(\d+)_earlystop_min(\d+)_pati(\d+)$")
SPECIES_ABBREVIATIONS = {
    "Aedes aegypti": "Ae.aeg",
    "Aedes albopictus": "Ae.alb",
    "Culex quinquefasciatus": "Cx.qui",
    "Anopheles gambiae": "An.gam",
    "Anopheles arabiensis": "An.ara",
    "Anopheles dirus": "An.dir",
    "Culex pipiens": "Cx.pip",
    "Anopheles minimus": "An.min",
    "Anopheles stephensi": "An.ste",
}


def load_json_file(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_rows(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def expected_run_context_for_dir(config: Dict, run_dir_name: str) -> Dict:
    run_config = dict(config)
    run_config["seed"] = extract_seed_from_run_dir(run_dir_name)
    run_config["experiment_name"] = run_dir_name
    return run_context_payload(run_config)


def feature_context_matches(actual: Dict, expected: Dict) -> bool:
    keys = [
        "training_feature_signature",
        "validation_feature_signature",
        "test_feature_signature",
    ]
    return all(actual.get(key) == expected.get(key) for key in keys)


def list_run_dirs(outputs_root: Path, config: Dict = None) -> List[Path]:
    run_dirs = []
    for path in outputs_root.iterdir():
        if not path.is_dir():
            continue
        if RUN_DIR_PATTERN.match(path.name) or path.name.startswith("baseline_MTRCNN_seed"):
            if config is not None:
                run_context_path = path / "run_context.json"
                if not run_context_path.exists():
                    continue
                try:
                    run_context = load_json_file(run_context_path)
                except json.JSONDecodeError:
                    continue
                if not feature_context_matches(run_context, expected_run_context_for_dir(config, path.name)):
                    continue
            run_dirs.append(path)
    return sorted(run_dirs, key=lambda path: int(extract_seed_from_run_dir(path.name)))


def extract_seed_from_run_dir(run_dir_name: str) -> int:
    match = RUN_DIR_PATTERN.match(run_dir_name)
    if match:
        return int(match.group(1))
    legacy_match = re.search(r"seed(\d+)", run_dir_name)
    if legacy_match:
        return int(legacy_match.group(1))
    raise ValueError(f"Cannot parse seed from run directory: {run_dir_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible report assets from saved outputs.")
    parser.add_argument("--config", type=str, default="configs/default_experiment.json")
    parser.add_argument("--outputs-root", type=str, default="outputs")
    parser.add_argument("--feature-root", type=str, default="data/feature")
    parser.add_argument("--out-dir", type=str, default="./technical_report_assets")
    return parser.parse_args()


def save_json(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: List[float]) -> Dict[str, float]:
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return {"mean": mean_value, "std": math.sqrt(variance)}


def load_feature_payload(path: Path) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_split_payloads(feature_root: Path) -> Dict[str, Dict]:
    return {
        split: load_feature_payload(feature_root / f"{split}_features.pkl")
        for split in ["training", "validation", "test"]
    }


def dataset_statistics(feature_root: Path, sample_rate: int, hop_length: int) -> List[Dict]:
    rows = []
    seconds_per_frame = hop_length / sample_rate
    payloads = load_split_payloads(feature_root)
    for split in ["training", "validation", "test"]:
        payload = payloads[split]
        lengths = [int(item["feature"].shape[0]) for item in payload["items"]]
        counts = Counter(lengths)
        mode_frames, mode_count = counts.most_common(1)[0]
        rows.append(
            {
                "split": split,
                "num_clips": len(lengths),
                "mean_frames": round(sum(lengths) / len(lengths), 3),
                "median_frames": float(statistics.median(lengths)),
                "mode_frames": mode_frames,
                "mode_count": mode_count,
                "min_frames": min(lengths),
                "max_frames": max(lengths),
                "mean_seconds": round((sum(lengths) / len(lengths)) * seconds_per_frame, 3),
                "median_seconds": round(float(statistics.median(lengths)) * seconds_per_frame, 3),
                "mode_seconds": round(mode_frames * seconds_per_frame, 3),
                "min_seconds": round(min(lengths) * seconds_per_frame, 3),
                "max_seconds": round(max(lengths) * seconds_per_frame, 3),
                "iterations_at_batch_64": math.ceil(len(lengths) / 64),
            }
        )
    return rows


def class_distribution_tables(feature_root: Path) -> Tuple[List[Dict], List[Dict]]:
    payloads = load_split_payloads(feature_root)
    species_counter = defaultdict(lambda: {"training": 0, "validation": 0, "test": 0})
    domain_counter = defaultdict(lambda: {"training": 0, "validation": 0, "test": 0})

    for split, payload in payloads.items():
        for item in payload["items"]:
            species_counter[item["species"]][split] += 1
            domain_counter[item["domain"]][split] += 1

    species_rows = []
    for species_name in sorted(species_counter):
        row = {"species": species_name}
        row.update(species_counter[species_name])
        species_rows.append(row)

    domain_rows = []
    for domain_name in sorted(domain_counter):
        row = {"domain": domain_name}
        row.update(domain_counter[domain_name])
        domain_rows.append(row)
    return species_rows, domain_rows


def model_summary(config: Dict) -> Dict:
    model = build_model(config, torch.device("cpu"))
    total_parameters = sum(param.numel() for param in model.parameters())
    trainable_parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "model_name": model.__class__.__name__,
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
    }


def load_metrics_csv(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json_any(paths: List[Path]) -> Dict:
    for path in paths:
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"None of the expected files exist: {[str(path) for path in paths]}")


def run_statistics(outputs_root: Path, config: Dict) -> List[Dict]:
    rows = []
    run_dirs = list_run_dirs(outputs_root, config)
    for run_dir in run_dirs:
        metrics_rows = load_metrics_csv(run_dir / "metrics.csv")
        best_row = max(metrics_rows, key=lambda row: float(row["val_species_balanced_accuracy"]))
        val_metrics = load_json_any(
            [run_dir / "best_model_eval" / "validation_metrics.json", run_dir / "model_best" / "validation_metrics.json", run_dir / "val_metrics.json"]
        )
        final_val_metrics = load_json_any(
            [run_dir / "final_model_eval" / "validation_metrics.json", run_dir / "model_final" / "validation_metrics.json", run_dir / "val_metrics_final.json"]
        )
        test_metrics = load_json_any(
            [run_dir / "best_model_eval" / "test_metrics.json", run_dir / "model_best" / "test_metrics.json", run_dir / "test_metrics.json"]
        )
        rows.append(
            {
                "seed": extract_seed_from_run_dir(run_dir.name),
                "run_dir": str(run_dir),
                "epochs_ran": len(metrics_rows),
                "best_epoch": int(best_row["epoch"]),
                "best_val_species_ba": float(best_row["val_species_balanced_accuracy"]),
                "final_val_species_ba": float(final_val_metrics["species_balanced_accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_species_accuracy": float(val_metrics["species_accuracy"]),
                "val_species_balanced_accuracy": float(val_metrics["species_balanced_accuracy"]),
                "val_domain_accuracy": float(val_metrics["domain_accuracy"]),
                "test_species_accuracy": float(test_metrics["species_accuracy"]),
                "test_species_balanced_accuracy": float(test_metrics["species_balanced_accuracy"]),
                "test_BA_seen": None if test_metrics.get("BA_seen") is None else float(test_metrics["BA_seen"]),
                "test_BA_unseen": None if test_metrics.get("BA_unseen") is None else float(test_metrics["BA_unseen"]),
                "test_DSG": None if test_metrics.get("DSG") is None else float(test_metrics["DSG"]),
                "test_domain_accuracy": float(test_metrics["domain_accuracy"]),
                "test_loss": float(test_metrics["loss"]),
            }
        )
    return rows


def aggregate_metric_rows(run_rows: List[Dict]) -> List[Dict]:
    metric_map = {
        "species_accuracy": ("val_species_accuracy", "test_species_accuracy"),
        "species_balanced_accuracy": ("val_species_balanced_accuracy", "test_species_balanced_accuracy"),
        "domain_accuracy": ("val_domain_accuracy", "test_domain_accuracy"),
        "loss": ("val_loss", "test_loss"),
        "epochs_ran": ("epochs_ran", "epochs_ran"),
    }
    rows = []
    for metric_name, (val_key, test_key) in metric_map.items():
        val_stats = mean_std([float(row[val_key]) for row in run_rows])
        test_stats = mean_std([float(row[test_key]) for row in run_rows])
        rows.append(
            {
                "metric": metric_name,
                "validation_mean": round(val_stats["mean"], 6),
                "validation_std": round(val_stats["std"], 6),
                "test_mean": round(test_stats["mean"], 6),
                "test_std": round(test_stats["std"], 6),
            }
        )
    for metric_name in ["test_BA_seen", "test_BA_unseen", "test_DSG"]:
        values = [float(row[metric_name]) for row in run_rows if row[metric_name] is not None]
        if not values:
            continue
        test_stats = mean_std(values)
        rows.append(
            {
                "metric": metric_name.replace("test_", ""),
                "validation_mean": "n/a",
                "validation_std": "n/a",
                "test_mean": round(test_stats["mean"], 6),
                "test_std": round(test_stats["std"], 6),
            }
        )
    return rows


def per_domain_statistics(outputs_root: Path, config: Dict) -> List[Dict]:
    bucket = defaultdict(list)
    for run_dir in list_run_dirs(outputs_root, config):
        metrics = load_json_any(
            [run_dir / "best_model_eval" / "test_metrics.json", run_dir / "model_best" / "test_metrics.json", run_dir / "test_metrics.json"]
        )
        for key, value in metrics.items():
            if key.startswith("species_ba_D"):
                bucket[key.replace("species_ba_", "")].append(float(value))

    rows = []
    for domain_name in sorted(bucket):
        stats = mean_std(bucket[domain_name])
        rows.append(
            {
                "domain": domain_name,
                "mean_test_species_ba": round(stats["mean"], 6),
                "std_test_species_ba": round(stats["std"], 6),
            }
        )
    return rows


def official_prediction_path(run_dir: Path, model_name: str) -> Path:
    return run_dir / model_name / "test_predictions.jsonl"


def species_recall(rows: List[Dict], species_name: str) -> float:
    subset = [row for row in rows if row["true_species_label"] == species_name]
    if not subset:
        return None
    correct = sum(1 for row in subset if row["predicted_species_label"] == species_name)
    return correct / len(subset)


def per_species_official_statistics(outputs_root: Path, config: Dict, model_name: str) -> List[Dict]:
    bucket = {
        species_name: {"BA_seen": [], "BA_unseen": [], "DSG": []}
        for species_name in SPECIES_NAMES
    }

    for run_dir in list_run_dirs(outputs_root, config):
        rows = load_jsonl_rows(official_prediction_path(run_dir, model_name))
        for species_name in SPECIES_NAMES:
            seen_rows = [
                row for row in rows
                if row.get("true_species_label") == species_name and row.get("evaluation_partition") == "seen"
            ]
            unseen_rows = [
                row for row in rows
                if row.get("true_species_label") == species_name and row.get("evaluation_partition") == "unseen"
            ]
            ba_seen = species_recall(seen_rows, species_name)
            ba_unseen = species_recall(unseen_rows, species_name)
            if ba_seen is not None:
                bucket[species_name]["BA_seen"].append(ba_seen)
            if ba_unseen is not None:
                bucket[species_name]["BA_unseen"].append(ba_unseen)
            if ba_seen is not None and ba_unseen is not None:
                bucket[species_name]["DSG"].append(abs(ba_unseen - ba_seen))

    rows = []
    for species_name in SPECIES_NAMES:
        row = {
            "species": species_name,
            "species_short": SPECIES_ABBREVIATIONS[species_name],
        }
        for metric_name in ["BA_seen", "BA_unseen", "DSG"]:
            values = bucket[species_name][metric_name]
            if values:
                stats = mean_std(values)
                row[f"{metric_name}_mean"] = round(stats["mean"], 6)
                row[f"{metric_name}_std"] = round(stats["std"], 6)
            else:
                row[f"{metric_name}_mean"] = None
                row[f"{metric_name}_std"] = None
        rows.append(row)
    return rows


def per_seed_official_summary(run_rows: List[Dict]) -> List[Dict]:
    rows = []
    for row in run_rows:
        rows.append(
            {
                "seed": row["seed"],
                "epochs_ran": row["epochs_ran"],
                "best_epoch": row["best_epoch"],
                "best_model_test_species_accuracy": round(row["test_species_accuracy"], 6),
                "best_model_test_species_balanced_accuracy": round(row["test_species_balanced_accuracy"], 6),
                "best_model_BA_seen": None if row["test_BA_seen"] is None else round(row["test_BA_seen"], 6),
                "best_model_BA_unseen": None if row["test_BA_unseen"] is None else round(row["test_BA_unseen"], 6),
                "best_model_DSG": None if row["test_DSG"] is None else round(row["test_DSG"], 6),
            }
        )
    return rows


def official_species_counts(config: Dict) -> List[Dict]:
    split_summary_path = Path(config["train_ids_path"]).parent / "split_summary.json"
    summary_payload = load_json_file(split_summary_path)
    unseen_domain_by_species = summary_payload["unseen_domain_by_species"]
    test_ids_path = Path(config["test_ids_path"])

    counts = {
        species_name: {"seen_test_clips": 0, "unseen_test_clips": 0}
        for species_name in SPECIES_NAMES
    }
    with open(test_ids_path, "r", encoding="utf-8") as handle:
        for line in handle:
            file_id = line.strip()
            if not file_id:
                continue
            species_name, domain_name = parse_file_id(file_id)
            if unseen_domain_by_species[species_name] == domain_name:
                counts[species_name]["unseen_test_clips"] += 1
            else:
                counts[species_name]["seen_test_clips"] += 1

    rows = []
    for species_name in SPECIES_NAMES:
        rows.append(
            {
                "species": species_name,
                "species_short": SPECIES_ABBREVIATIONS[species_name],
                "seen_test_clips": counts[species_name]["seen_test_clips"],
                "unseen_test_clips": counts[species_name]["unseen_test_clips"],
            }
        )
    return rows


def best_seed_summary(run_rows: List[Dict]) -> Dict:
    return max(run_rows, key=lambda row: row["test_species_balanced_accuracy"])


def plot_epochs(run_rows: List[Dict], out_path: Path) -> None:
    seeds = [str(row["seed"]) for row in run_rows]
    epochs_ran = [row["epochs_ran"] for row in run_rows]
    best_epochs = [row["best_epoch"] for row in run_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(seeds))
    ax.bar(x, epochs_ran, label="epochs_ran", color="#6baed6")
    ax.scatter(x, best_epochs, label="best_epoch", color="#de2d26", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(seeds, rotation=45)
    ax.set_ylabel("Epoch")
    ax.set_title("Training Length And Best Epoch Per Seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_species_ba(run_rows: List[Dict], out_path: Path) -> None:
    seeds = [str(row["seed"]) for row in run_rows]
    val_scores = [row["val_species_balanced_accuracy"] for row in run_rows]
    test_scores = [row["test_species_balanced_accuracy"] for row in run_rows]
    x = list(range(len(seeds)))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.38
    ax.bar([i - width / 2 for i in x], val_scores, width=width, label="validation", color="#31a354")
    ax.bar([i + width / 2 for i in x], test_scores, width=width, label="test", color="#756bb1")
    ax.set_xticks(x)
    ax.set_xticklabels(seeds, rotation=45)
    ax.set_ylabel("Species Balanced Accuracy")
    ax.set_title("Validation And Test Species Balanced Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_domain(domain_rows: List[Dict], out_path: Path) -> None:
    domains = [row["domain"] for row in domain_rows]
    means = [row["mean_test_species_ba"] for row in domain_rows]
    stds = [row["std_test_species_ba"] for row in domain_rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(domains, means, yerr=stds, capsize=4, color="#fd8d3c")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Test Species Balanced Accuracy")
    ax.set_title("Per-Domain Test Species Balanced Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_per_species_official(per_species_rows: List[Dict], out_path: Path) -> None:
    species = [row["species_short"] for row in per_species_rows]
    seen_means = [float("nan") if row["BA_seen_mean"] is None else row["BA_seen_mean"] for row in per_species_rows]
    seen_stds = [0.0 if row["BA_seen_std"] is None else row["BA_seen_std"] for row in per_species_rows]
    unseen_means = [float("nan") if row["BA_unseen_mean"] is None else row["BA_unseen_mean"] for row in per_species_rows]
    unseen_stds = [0.0 if row["BA_unseen_std"] is None else row["BA_unseen_std"] for row in per_species_rows]

    x = list(range(len(species)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar([i - width / 2 for i in x], seen_means, width=width, yerr=seen_stds, capsize=4, label="Seen", color="#4c78a8")
    ax.bar([i + width / 2 for i in x], unseen_means, width=width, yerr=unseen_stds, capsize=4, label="Unseen", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(species)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Per-Species Official Evaluation: Seen vs Unseen")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_validation_curves(run_dirs: List[Path], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for run_dir in run_dirs:
        rows = load_metrics_csv(run_dir / "metrics.csv")
        epochs = [int(row["epoch"]) for row in rows]
        scores = [float(row["val_species_balanced_accuracy"]) for row in rows]
        ax.plot(epochs, scores, linewidth=1.5, label=str(extract_seed_from_run_dir(run_dir.name)))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Species Balanced Accuracy")
    ax.set_title("Validation Species Balanced Accuracy Curves")
    ax.legend(title="Seed", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_loss_curves(outputs_root: Path, run_rows: List[Dict], out_path: Path) -> None:
    best_run = best_seed_summary(run_rows)
    rows = load_metrics_csv(Path(best_run["run_dir"]) / "metrics.csv")
    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_loss, label="train_loss", color="#3182bd", linewidth=2)
    ax.plot(epochs, val_loss, label="val_loss", color="#e6550d", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Train/Validation Loss Curves (Best Seed {best_run['seed']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_accuracy_curves(outputs_root: Path, run_rows: List[Dict], out_path: Path) -> None:
    best_run = best_seed_summary(run_rows)
    rows = load_metrics_csv(Path(best_run["run_dir"]) / "metrics.csv")
    epochs = [int(row["epoch"]) for row in rows]
    train_acc = [float(row["train_species_accuracy"]) for row in rows]
    val_acc = [float(row["val_species_accuracy"]) for row in rows]
    val_ba = [float(row["val_species_balanced_accuracy"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, train_acc, label="train_species_accuracy", color="#31a354", linewidth=2)
    ax.plot(epochs, val_acc, label="val_species_accuracy", color="#756bb1", linewidth=2)
    ax.plot(epochs, val_ba, label="val_species_balanced_accuracy", color="#de2d26", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Accuracy Curves (Best Seed {best_run['seed']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_length_histograms(feature_root: Path, out_path: Path) -> None:
    payloads = load_split_payloads(feature_root)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, split in zip(axes, ["training", "validation", "test"]):
        lengths = [int(item["feature"].shape[0]) for item in payloads[split]["items"]]
        capped = [min(length, 500) for length in lengths]
        ax.hist(capped, bins=50, color="#6baed6")
        ax.set_title(split)
        ax.set_xlabel("Frames (capped at 500)")
    axes[0].set_ylabel("Clip count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_species_distribution(species_rows: List[Dict], out_path: Path) -> None:
    species_names = [row["species"] for row in species_rows]
    training = [row["training"] for row in species_rows]
    validation = [row["validation"] for row in species_rows]
    test = [row["test"] for row in species_rows]
    x = list(range(len(species_names)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width for i in x], training, width=width, label="training", color="#31a354")
    ax.bar(x, validation, width=width, label="validation", color="#fd8d3c")
    ax.bar([i + width for i in x], test, width=width, label="test", color="#756bb1")
    ax.set_xticks(x)
    ax.set_xticklabels(species_names, rotation=45, ha="right")
    ax.set_ylabel("Clip count")
    ax.set_title("Species Distribution Per Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_domain_distribution(domain_rows: List[Dict], out_path: Path) -> None:
    domain_names = [row["domain"] for row in domain_rows]
    training = [row["training"] for row in domain_rows]
    validation = [row["validation"] for row in domain_rows]
    test = [row["test"] for row in domain_rows]
    x = list(range(len(domain_names)))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - width for i in x], training, width=width, label="training", color="#31a354")
    ax.bar(x, validation, width=width, label="validation", color="#fd8d3c")
    ax.bar([i + width for i in x], test, width=width, label="test", color="#756bb1")
    ax.set_xticks(x)
    ax.set_xticklabels(domain_names)
    ax.set_ylabel("Clip count")
    ax.set_title("Domain Distribution Per Split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_val_test_scatter(run_rows: List[Dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    val_scores = [row["val_species_balanced_accuracy"] for row in run_rows]
    test_scores = [row["test_species_balanced_accuracy"] for row in run_rows]
    for row in run_rows:
        ax.scatter(row["val_species_balanced_accuracy"], row["test_species_balanced_accuracy"], color="#3182bd")
        ax.text(
            row["val_species_balanced_accuracy"] + 0.0005,
            row["test_species_balanced_accuracy"] + 0.0005,
            str(row["seed"]),
            fontsize=8,
        )
    low = min(val_scores + test_scores) - 0.01
    high = max(val_scores + test_scores) + 0.01
    ax.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("Validation Species Balanced Accuracy")
    ax.set_ylabel("Test Species Balanced Accuracy")
    ax.set_title("Validation vs Test Species Balanced Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    config: Dict,
    model_info: Dict,
    dataset_rows: List[Dict],
    species_rows: List[Dict],
    domain_distribution_rows: List[Dict],
    run_rows: List[Dict],
    aggregate_rows: List[Dict],
    domain_rows: List[Dict],
    official_count_rows: List[Dict],
    best_model_species_rows: List[Dict],
    final_model_species_rows: List[Dict],
) -> None:
    lines = [
        "# Technical Report Reproducibility Summary",
        "",
        "## Configuration",
        "",
        f"- sample_rate: {config['sample_rate']}",
        f"- n_fft: {config['n_fft']}",
        f"- hop_length: {config['hop_length']}",
        f"- win_length: {config['win_length']}",
        f"- n_mels: {config['n_mels']}",
        f"- batch_size: {config['batch_size']}",
        f"- eval_batch_size: {config.get('eval_batch_size', config['batch_size'])}",
        f"- train_crop_seconds: {config.get('train_crop_seconds')}",
        f"- learning_rate: {config['learning_rate']}",
        f"- frequency_resolution_hz: {config['sample_rate'] / config['n_fft']:.6f}",
        f"- model_name: {model_info['model_name']}",
        f"- total_parameters: {model_info['total_parameters']}",
        f"- trainable_parameters: {model_info['trainable_parameters']}",
        "",
        "## Dataset Statistics",
        "",
        "| Split | Num clips | Mean frames | Median frames | Mode frames | Min frames | Max frames | Mean sec | Max sec | Iterations at batch 64 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_rows:
        lines.append(
            f"| {row['split']} | {row['num_clips']} | {row['mean_frames']} | {row['median_frames']} | "
            f"{row['mode_frames']} | {row['min_frames']} | {row['max_frames']} | "
            f"{row['mean_seconds']} | {row['max_seconds']} | {row['iterations_at_batch_64']} |"
        )

    lines.extend(
        [
            "",
            "## Species Distribution Per Split",
            "",
            "| Species | Training | Validation | Test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in species_rows:
        lines.append(f"| {row['species']} | {row['training']} | {row['validation']} | {row['test']} |")

    lines.extend(
        [
            "",
            "## Domain Distribution Per Split",
            "",
            "| Domain | Training | Validation | Test |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in domain_distribution_rows:
        lines.append(f"| {row['domain']} | {row['training']} | {row['validation']} | {row['test']} |")

    lines.extend(
        [
            "",
            "## Official Seen/Unseen Test Counts",
            "",
            "| Species | Seen test clips | Unseen test clips |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in official_count_rows:
        lines.append(f"| {row['species_short']} | {row['seen_test_clips']} | {row['unseen_test_clips']} |")

    if run_rows:
        best_run = max(run_rows, key=lambda row: row["test_species_balanced_accuracy"])
        worst_run = min(run_rows, key=lambda row: row["test_species_balanced_accuracy"])

        lines.extend(
            [
                "",
                "## Multi-Seed Summary",
                "",
                "| Metric | Validation mean | Validation std | Test mean | Test std |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in aggregate_rows:
            lines.append(
                f"| {row['metric']} | {row['validation_mean']} | {row['validation_std']} | "
                f"{row['test_mean']} | {row['test_std']} |"
            )

        lines.extend(
            [
                "",
                "## Per-Domain Test Species Balanced Accuracy",
                "",
                "| Domain | Mean | Std |",
                "| --- | ---: | ---: |",
            ]
        )
        for row in domain_rows:
            lines.append(f"| {row['domain']} | {row['mean_test_species_ba']} | {row['std_test_species_ba']} |")

        lines.extend(
            [
                "",
                "## Per-Species Official Results (Best Model)",
                "",
                "| Species | BA_seen mean | BA_seen std | BA_unseen mean | BA_unseen std | DSG mean | DSG std |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in best_model_species_rows:
            lines.append(
                f"| {row['species_short']} | {row['BA_seen_mean']} | {row['BA_seen_std']} | "
                f"{row['BA_unseen_mean']} | {row['BA_unseen_std']} | {row['DSG_mean']} | {row['DSG_std']} |"
            )

        lines.extend(
            [
                "",
                "## Per-Species Official Results (Final Model)",
                "",
                "| Species | BA_seen mean | BA_seen std | BA_unseen mean | BA_unseen std | DSG mean | DSG std |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in final_model_species_rows:
            lines.append(
                f"| {row['species_short']} | {row['BA_seen_mean']} | {row['BA_seen_std']} | "
                f"{row['BA_unseen_mean']} | {row['BA_unseen_std']} | {row['DSG_mean']} | {row['DSG_std']} |"
            )

        lines.extend(
            [
                "",
                "## Seed-Level Training Summary",
                "",
                "| Seed | Epochs ran | Best epoch | Best val species BA | Final val species BA | Test species BA |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in run_rows:
            lines.append(
                f"| {row['seed']} | {row['epochs_ran']} | {row['best_epoch']} | "
                f"{row['best_val_species_ba']:.6f} | {row['final_val_species_ba']:.6f} | "
                f"{row['test_species_balanced_accuracy']:.6f} |"
            )

        lines.extend(
            [
                "",
                "## Seed-Level Notes",
                "",
                f"- best test species balanced accuracy: seed {best_run['seed']} -> {best_run['test_species_balanced_accuracy']:.6f}",
                f"- worst test species balanced accuracy: seed {worst_run['seed']} -> {worst_run['test_species_balanced_accuracy']:.6f}",
                f"- average epochs actually run: {mean_std([row['epochs_ran'] for row in run_rows])['mean']:.2f}",
                f"- average gap between best validation species BA and final validation species BA: "
                f"{mean_std([row['best_val_species_ba'] - row['final_val_species_ba'] for row in run_rows])['mean']:.6f}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Multi-Seed Summary",
                "",
                "No compatible run directories were found for the current config and metadata split under `outputs/`.",
                "Regenerate features first if needed, then retrain before using this script for result tables and plots.",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommended Report Items",
            "",
            "- dataset split sizes and feature-length statistics",
            "- species and domain distribution per split",
            "- feature extraction settings and normalization procedure",
            "- frequency resolution and mel frontend definition",
            "- model architecture, parameter count, and dual-head training objective",
            "- training setup, early stopping rule, optimizer, batch sizes, and device policy",
            "- train/validation loss curves and validation metric curves",
            "- validation and test metrics over all seeds",
            "- seed-level epoch counts, best epoch, and checkpoint selection rule",
            "- per-domain species balanced accuracy tables",
            "- saved prediction files for additional reproducible metrics",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    outputs_root = Path(args.outputs_root)
    feature_root = Path(args.feature_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = list_run_dirs(outputs_root, config)
    model_info = model_summary(config)
    dataset_rows = dataset_statistics(feature_root, config["sample_rate"], config["hop_length"])
    species_rows, domain_distribution_rows = class_distribution_tables(feature_root)
    official_count_rows = official_species_counts(config)
    run_rows = run_statistics(outputs_root, config)
    aggregate_rows = aggregate_metric_rows(run_rows) if run_rows else []
    domain_rows = per_domain_statistics(outputs_root, config) if run_rows else []
    best_model_species_rows = per_species_official_statistics(outputs_root, config, "best_model_eval") if run_rows else []
    final_model_species_rows = per_species_official_statistics(outputs_root, config, "final_model_eval") if run_rows else []
    per_seed_rows = per_seed_official_summary(run_rows) if run_rows else []

    write_csv(out_dir / "dataset_statistics.csv", dataset_rows)
    write_csv(out_dir / "species_distribution.csv", species_rows)
    write_csv(out_dir / "domain_distribution.csv", domain_distribution_rows)
    write_csv(out_dir / "official_species_counts.csv", official_count_rows)
    if run_rows:
        write_csv(out_dir / "run_statistics.csv", run_rows)
        write_csv(out_dir / "aggregate_metrics.csv", aggregate_rows)
        write_csv(out_dir / "per_domain_species_ba.csv", domain_rows)
        write_csv(out_dir / "per_species_official_best_model.csv", best_model_species_rows)
        write_csv(out_dir / "per_species_official_final_model.csv", final_model_species_rows)
        write_csv(out_dir / "per_seed_official_summary_best_model.csv", per_seed_rows)

    save_json(out_dir / "model_summary.json", model_info)
    save_json(out_dir / "dataset_statistics.json", dataset_rows)
    save_json(out_dir / "species_distribution.json", species_rows)
    save_json(out_dir / "domain_distribution.json", domain_distribution_rows)
    save_json(out_dir / "official_species_counts.json", official_count_rows)
    if run_rows:
        save_json(out_dir / "run_statistics.json", run_rows)
        save_json(out_dir / "aggregate_metrics.json", aggregate_rows)
        save_json(out_dir / "per_domain_species_ba.json", domain_rows)
        save_json(out_dir / "per_species_official_best_model.json", best_model_species_rows)
        save_json(out_dir / "per_species_official_final_model.json", final_model_species_rows)
        save_json(out_dir / "per_seed_official_summary_best_model.json", per_seed_rows)

        plot_epochs(run_rows, out_dir / "epochs_per_seed.png")
        plot_species_ba(run_rows, out_dir / "species_ba_per_seed.png")
        plot_per_domain(domain_rows, out_dir / "per_domain_species_ba.png")
        plot_per_species_official(best_model_species_rows, out_dir / "per_species_official_best_model.png")
        plot_validation_curves(run_dirs, out_dir / "validation_species_ba_curves.png")
        plot_loss_curves(outputs_root, run_rows, out_dir / "loss_curves_best_seed.png")
        plot_accuracy_curves(outputs_root, run_rows, out_dir / "accuracy_curves_best_seed.png")
    plot_length_histograms(feature_root, out_dir / "feature_length_histograms.png")
    plot_species_distribution(species_rows, out_dir / "species_distribution.png")
    plot_domain_distribution(domain_distribution_rows, out_dir / "domain_distribution.png")
    if run_rows:
        plot_val_test_scatter(run_rows, out_dir / "val_test_species_ba_scatter.png")

    write_markdown_report(
        out_dir / "technical_report_summary.md",
        config,
        model_info,
        dataset_rows,
        species_rows,
        domain_distribution_rows,
        run_rows,
        aggregate_rows,
        domain_rows,
        official_count_rows,
        best_model_species_rows,
        final_model_species_rows,
    )

    print(f"Saved technical report assets to {out_dir}")


if __name__ == "__main__":
    main()
