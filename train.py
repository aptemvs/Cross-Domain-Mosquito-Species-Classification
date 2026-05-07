"""Training entry point

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW

from framework.config import (
    load_config,
    run_context_payload,
)
from framework.dataset import get_loader
from framework.engine import evaluate_model, train_one_epoch
from framework.git import get_git_commit_summary
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import (
    acquire_experiment_lock,
    append_metrics,
    build_model,
    choose_device,
    load_json,
    make_logger,
    make_output_dir,
    save_json,
    set_seed,
    max_train_frames,
    release_experiment_lock,
)

from framework.utilization import generate_trials
from schema.trial import TrialConfig
from evaluate import evaluate_checkpoint, save_prediction_rows
from const.enum import Split

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mosquito classifier.")
    parser.add_argument("--config", type=str, default="configs/default_experiment.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def trial_name(trial: TrialConfig) -> str:
    seed = trial.seed
    batch_size = trial.batch_size
    epochs = trial.epochs
    min_epoch = trial.early_stopping_min_epoch
    patience = trial.early_stopping_patience
    model_name = trial.backend.model.value

    return f"{model_name}_seed{seed}_B{batch_size}_E{epochs}_earlystop_min{min_epoch}_pati{patience}"


def evaluate_and_save_outputs(config: TrialConfig, checkpoint_path: Path, output_dir: Path, model_name: str) -> dict:

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    validation_result = evaluate_checkpoint(config, checkpoint_path, Split.VALIDATION, return_predictions=True)
    save_json(model_output_dir / "validation_metrics.json", validation_result["metrics"])
    save_prediction_rows(model_output_dir / "validation_predictions.jsonl", validation_result["predictions"])

    test_result = evaluate_checkpoint(config, checkpoint_path, Split.TEST, return_predictions=True)
    save_json(model_output_dir / "test_metrics.json", test_result["metrics"])
    save_prediction_rows(model_output_dir / "test_predictions.jsonl", test_result["predictions"])

    return {
        "output_dir": str(model_output_dir),
        "validation_metrics": validation_result["metrics"],
        "test_metrics": test_result["metrics"],
        "validation_metrics_path": str(model_output_dir / "validation_metrics.json"),
        "validation_predictions_path": str(model_output_dir / "validation_predictions.jsonl"),
        "test_metrics_path": str(model_output_dir / "test_metrics.json"),
        "test_predictions_path": str(model_output_dir / "test_predictions.jsonl"),
    }


def train_experiment(trial_config: TrialConfig, overwrite: bool = False) -> dict:
    experiment_name = trial_name(trial_config)
    set_seed(trial_config.seed)
    commit_summary = get_git_commit_summary()
    device = choose_device(trial_config.device)
    output_dir = make_output_dir(trial_config.output_root, experiment_name)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_context_path = output_dir / "run_context.json"
    best_checkpoint_path = model_dir / "model_best.pth"
    final_checkpoint_path = model_dir / "model_final.pth"
    best_eval_dir = output_dir / "best_model_eval"
    final_eval_dir = output_dir / "final_model_eval"
    required_eval_files = [
        best_eval_dir / "validation_metrics.json",
        best_eval_dir / "validation_predictions.jsonl",
        best_eval_dir / "test_metrics.json",
        best_eval_dir / "test_predictions.jsonl",
        final_eval_dir / "validation_metrics.json",
        final_eval_dir / "validation_predictions.jsonl",
        final_eval_dir / "test_metrics.json",
        final_eval_dir / "test_predictions.jsonl",
    ]
    current_run_context = run_context_payload(trial_config)

    if (
        run_context_path.exists()
        and best_checkpoint_path.exists()
        and final_checkpoint_path.exists()
        and all(path.exists() for path in required_eval_files)
        and load_json(run_context_path) == current_run_context
        and not overwrite
    ):
        print(f"loading from {best_checkpoint_path}")
        best_validation_metrics = load_json(best_eval_dir / "validation_metrics.json")
        best_test_metrics = load_json(best_eval_dir / "test_metrics.json")
        final_validation_metrics = load_json(final_eval_dir / "validation_metrics.json")
        final_test_metrics = load_json(final_eval_dir / "test_metrics.json")
        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "best_eval": {
                "output_dir": str(best_eval_dir),
                "validation_metrics": best_validation_metrics,
                "test_metrics": best_test_metrics,
                "validation_metrics_path": str(
                    best_eval_dir / "validation_metrics.json"
                ),
                "validation_predictions_path": str(
                    best_eval_dir / "validation_predictions.jsonl"
                ),
                "test_metrics_path": str(best_eval_dir / "test_metrics.json"),
                "test_predictions_path": str(best_eval_dir / "test_predictions.jsonl"),
            },
            "final_eval": {
                "output_dir": str(final_eval_dir),
                "validation_metrics": final_validation_metrics,
                "test_metrics": final_test_metrics,
                "validation_metrics_path": str(
                    final_eval_dir / "validation_metrics.json"
                ),
                "validation_predictions_path": str(
                    final_eval_dir / "validation_predictions.jsonl"
                ),
                "test_metrics_path": str(final_eval_dir / "test_metrics.json"),
                "test_predictions_path": str(final_eval_dir / "test_predictions.jsonl"),
            },
        }

    lock_path = acquire_experiment_lock(output_dir, experiment_name)
    if lock_path is None:
        print(f"already running: {output_dir / '.experiment.lock'}")
        return {
            "status": "running",
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
        }
    print(f"training model to {output_dir}")
    logger = None
    try:
        wandb.init(
            entity="biodcase-2026-cd-msc",
            project="BioDCASE_Task5",
            group=commit_summary,
            name=f"{experiment_name}_{commit_summary}",
            config=trial_config.model_dump(mode="json"),
        )
        save_json(run_context_path, current_run_context)
        save_json(output_dir / "resolved_config.json", trial_config.model_dump(mode="json"))
        logger = make_logger(output_dir / "train.log")

        train_loader = get_loader(
            feature_root=trial_config.feature_root,
            split=Split.TRAINING,
            batch_size=trial_config.batch_size,
            num_workers=trial_config.num_workers,
            max_train_frames=max_train_frames(trial_config),
            training=True,
            shuffle=True,
            pin_memory=device.type == "cuda",
            config=trial_config,
            normalize_features=trial_config.normalize_features,
            verify_config_signature=True,
            verify_stats_signature=True,
        )
        val_loader = get_loader(
            feature_root=trial_config.feature_root,
            split=Split.VALIDATION,
            batch_size=trial_config.eval_batch_size,
            num_workers=trial_config.num_workers,
            max_train_frames=None,
            training=False,
            shuffle=False,
            pin_memory=device.type == "cuda",
            config=trial_config,
            normalize_features=trial_config.normalize_features,
            verify_config_signature=True,
            verify_stats_signature=True,
        )

        model = build_model(trial_config, device)
        optimizer = AdamW(model.parameters(), lr=trial_config.learning_rate, weight_decay=trial_config.weight_decay)
        early_stopping_min_epoch = trial_config.early_stopping_min_epoch
        early_stopping_patience = trial_config.early_stopping_patience

        best_score = float("-inf")
        best_epoch = 0
        best_val_metrics = {}
        last_val_metrics = {}
        epochs_without_improvement = 0
        epoch = 0

        def numeric_metrics_only(metrics: dict) -> dict:
            return {
                key: value
                for key, value in metrics.items()
                if value is None or isinstance(value, (int, float))
            }

        def with_prefix(prefix: str, metrics: dict) -> dict:
            return {f"{prefix}_{key}": value for key, value in metrics.items()}

        def rounded_prefixed_metrics(prefix: str, metrics: dict) -> dict:
            payload = {}
            for key, value in numeric_metrics_only(metrics).items():
                payload[f"{prefix}_{key}"] = round(value, 6) if value is not None else None
            return payload

        for epoch in range(1, trial_config.epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                num_species_classes=len(SPECIES_NAMES),
                num_domain_classes=len(DOMAIN_NAMES),
            )
            last_val_metrics = val_metrics

            row = {
                "epoch": epoch,
                **rounded_prefixed_metrics("train", train_metrics),
                **rounded_prefixed_metrics("val", val_metrics),
                "lr": optimizer.param_groups[0]["lr"],
            }
            for domain_name in DOMAIN_NAMES:
                val_species_ba = val_metrics.get(f"species_ba_{domain_name}")
                row[f"val_species_ba_{domain_name}"] = (
                    round(val_species_ba, 6) if val_species_ba is not None else None
                )

            append_metrics(output_dir / "metrics.csv", row)
            logger.info(row)
            wandb.log(numeric_metrics_only(row))

            current_score = val_metrics["species_balanced_accuracy"]
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": trial_config,
                        "epoch": epoch,
                        "val_metrics": best_val_metrics,
                        "selection_metric": "species_balanced_accuracy",
                    },
                    best_checkpoint_path,
                )
                logger.info("Saved best checkpoint to %s", best_checkpoint_path)
            elif epoch >= early_stopping_min_epoch:
                epochs_without_improvement += 1

            if epoch >= early_stopping_min_epoch and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %s. Best epoch: %s, best validation species_balanced_accuracy: %.6f, min_epoch: %s, patience: %s",
                    epoch,
                    best_epoch,
                    best_score,
                    early_stopping_min_epoch,
                    early_stopping_patience,
                )
                break

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": trial_config,
                "epoch": epoch,
                "val_metrics": last_val_metrics,
            },
            final_checkpoint_path,
        )
        logger.info("Saved final checkpoint to %s", final_checkpoint_path)

        logger.info("Evaluating best checkpoint outputs.")
        best_eval = evaluate_and_save_outputs(trial_config, best_checkpoint_path, output_dir, "best_model_eval")
        logger.info("Evaluating final checkpoint outputs.")
        final_eval = evaluate_and_save_outputs(trial_config, final_checkpoint_path, output_dir, "final_model_eval")

        wandb.log(
            {
                **with_prefix("best_val", numeric_metrics_only(best_eval["validation_metrics"])),
                **with_prefix("best_test", numeric_metrics_only(best_eval["test_metrics"])),
                **with_prefix("final_val", numeric_metrics_only(final_eval["validation_metrics"])),
                **with_prefix("final_test", numeric_metrics_only(final_eval["test_metrics"])),
            }
        )

        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "best_eval": best_eval,
            "final_eval": final_eval,
        }
    finally:
        if logger is not None:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)
        if wandb.run is not None:
            wandb.finish()
        release_experiment_lock(lock_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trials = list(generate_trials(config))
    if len(trials) > 1:
        raise ValueError(
            "Config describes more than one trial. Use run_multiple_experiments.py"
        )
    result = train_experiment(trials[0], overwrite=args.overwrite)
    if result["status"] == "running":
        return


if __name__ == "__main__":
    main()
