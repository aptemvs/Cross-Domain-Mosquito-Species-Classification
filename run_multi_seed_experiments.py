"""Run 10 fixed-seed experiments for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
from copy import deepcopy
from pathlib import Path

from framework.config import load_config, run_context_payload
from framework.utilization import format_mean_std, load_json, save_json, write_csv, write_summary_table
from train import experiment_name_for_seed, train_experiment

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 10 fixed-seed training and evaluation jobs.")
    parser.add_argument("--config", type=str, default="configs/multi_seed_experiment.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_run_config(base_config: dict, seed: int) -> dict:
    config = deepcopy(base_config)
    config["seed"] = seed
    config["experiment_name"] = experiment_name_for_seed(seed, config)
    return config


def normalize_run_summary(run_summary: dict) -> dict:
    if "best_model_eval" in run_summary and "final_model_eval" in run_summary:
        return run_summary
    if "model_best" in run_summary and "model_final" in run_summary:
        run_summary["best_model_eval"] = run_summary.pop("model_best")
        run_summary["final_model_eval"] = run_summary.pop("model_final")
    return run_summary


def summary_rows_for_model(run_summaries: list, model_name: str) -> list:
    rows = []
    for summary in run_summaries:
        metrics = summary[model_name]
        checkpoint_key = "best_checkpoint_path" if model_name == "best_model_eval" else "final_checkpoint_path"
        rows.append(
            {
                "seed": summary["seed"],
                "checkpoint_path": summary[checkpoint_key],
                "val_loss": round(metrics["validation_metrics"]["loss"], 6),
                "val_species_accuracy": round(metrics["validation_metrics"]["species_accuracy"], 6),
                "val_species_balanced_accuracy": round(metrics["validation_metrics"]["species_balanced_accuracy"], 6),
                "val_domain_accuracy": round(metrics["validation_metrics"]["domain_accuracy"], 6),
                "test_loss": round(metrics["test_metrics"]["loss"], 6),
                "test_species_accuracy": round(metrics["test_metrics"]["species_accuracy"], 6),
                "test_species_balanced_accuracy": round(metrics["test_metrics"]["species_balanced_accuracy"], 6),
                "test_BA_seen": None if metrics["test_metrics"].get("BA_seen") is None else round(metrics["test_metrics"]["BA_seen"], 6),
                "test_BA_unseen": None if metrics["test_metrics"].get("BA_unseen") is None else round(metrics["test_metrics"]["BA_unseen"], 6),
                "test_DSG": None if metrics["test_metrics"].get("DSG") is None else round(metrics["test_metrics"]["DSG"], 6),
                "test_num_seen_samples": metrics["test_metrics"].get("num_seen_samples"),
                "test_num_unseen_samples": metrics["test_metrics"].get("num_unseen_samples"),
                "test_domain_accuracy": round(metrics["test_metrics"]["domain_accuracy"], 6),
            }
        )
    return rows


def aggregate_report_rows(rows: list) -> list:
    report_rows = [
        {
            "metric": "loss",
            "validation": format_mean_std([row["val_loss"] for row in rows]),
            "test": format_mean_std([row["test_loss"] for row in rows]),
        },
        {
            "metric": "species_accuracy",
            "validation": format_mean_std([row["val_species_accuracy"] for row in rows]),
            "test": format_mean_std([row["test_species_accuracy"] for row in rows]),
        },
        {
            "metric": "species_balanced_accuracy",
            "validation": format_mean_std([row["val_species_balanced_accuracy"] for row in rows]),
            "test": format_mean_std([row["test_species_balanced_accuracy"] for row in rows]),
        },
        {
            "metric": "domain_accuracy",
            "validation": format_mean_std([row["val_domain_accuracy"] for row in rows]),
            "test": format_mean_std([row["test_domain_accuracy"] for row in rows]),
        },
    ]
    ba_seen_values = [row["test_BA_seen"] for row in rows if row["test_BA_seen"] is not None]
    ba_unseen_values = [row["test_BA_unseen"] for row in rows if row["test_BA_unseen"] is not None]
    dsg_values = [row["test_DSG"] for row in rows if row["test_DSG"] is not None]

    if ba_seen_values:
        report_rows.append(
            {
                "metric": "BA_seen",
                "validation": "n/a",
                "test": format_mean_std(ba_seen_values),
            }
        )
    if ba_unseen_values:
        report_rows.append(
            {
                "metric": "BA_unseen",
                "validation": "n/a",
                "test": format_mean_std(ba_unseen_values),
            }
        )
    if dsg_values:
        report_rows.append(
            {
                "metric": "DSG",
                "validation": "n/a",
                "test": format_mean_std(dsg_values),
            }
        )
    return report_rows


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    run_summaries = []
    skipped_runs = []

    for seed in base_config["seeds"]:
        config = build_run_config(base_config, seed)
        output_dir = Path(config["output_root"]) / config["experiment_name"]
        run_summary_path = output_dir / "run_summary.json"
        run_context_path = output_dir / "run_context.json"
        force_overwrite = args.overwrite
        current_run_context = run_context_payload(config)

        if (
            run_summary_path.exists()
            and run_context_path.exists()
            and load_json(run_context_path) == current_run_context
            and not force_overwrite
        ):
            run_summary = normalize_run_summary(load_json(run_summary_path))
            print(f"loading from {run_summary_path}")
            run_summaries.append(run_summary)
            continue

        if not run_summary_path.exists() or force_overwrite:
            train_result = train_experiment(config, overwrite=force_overwrite)
            if train_result["status"] == "running":
                message = f"skipping {config['experiment_name']}: already running"
                print(message)
                skipped_runs.append(
                    {
                        "seed": seed,
                        "experiment_name": config["experiment_name"],
                        "output_dir": str(output_dir),
                        "reason": "already running",
                    }
                )
                continue
            run_summary = {
                "seed": seed,
                "output_dir": train_result["output_dir"],
                "best_checkpoint_path": train_result["best_checkpoint_path"],
                "final_checkpoint_path": train_result["final_checkpoint_path"],
                "best_model_eval": train_result["best_eval"],
                "final_model_eval": train_result["final_eval"],
            }
            save_json(run_summary_path, run_summary)
            run_summaries.append(run_summary)

    report_dir = Path(base_config["output_root"]) / "multi_seed_summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    save_json(report_dir / "run_summaries.json", run_summaries)
    save_json(report_dir / "skipped_runs.json", skipped_runs)

    for model_name in ["best_model_eval", "final_model_eval"]:
        model_rows = summary_rows_for_model(run_summaries, model_name)
        if not model_rows:
            continue
        report_rows = aggregate_report_rows(model_rows)
        if model_name == "best_model_eval":
            model_report_dir = report_dir / "best_model_eval"
        else:
            model_report_dir = report_dir / "final_model_eval"
        model_report_dir.mkdir(parents=True, exist_ok=True)
        write_csv(model_report_dir / "all_runs.csv", model_rows)
        save_json(model_report_dir / "all_runs.json", model_rows)
        save_json(model_report_dir / "summary_stats.json", report_rows)
        write_summary_table(model_report_dir / "summary_report.md", report_rows)


if __name__ == "__main__":
    main()
