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
    config_signature,
    feature_signature_payload,
    load_config,
    run_context_payload,
)
from framework.dataset import MosquitoFeatureDataset, pad_collate_fn
from framework.engine import evaluate_model, train_one_epoch
from framework.git import get_commit_hash
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import (
    acquire_experiment_lock,
    append_metrics,
    build_model,
    choose_device,
    load_json,
    make_loader,
    make_logger,
    make_output_dir,
    save_json,
    set_seed,
    split_feature_path,
    training_stats_path,
    max_train_frames,
    release_experiment_lock,
)

from utils.generate_trials import generate_trials
from schema.trial_config import TrialConfig
from evaluate import evaluate_checkpoint, save_prediction_rows


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

    validation_result = evaluate_checkpoint(config, checkpoint_path, "validation", return_predictions=True)
    save_json(model_output_dir / "validation_metrics.json", validation_result["metrics"])
    save_prediction_rows(model_output_dir / "validation_predictions.jsonl", validation_result["predictions"])

    test_result = evaluate_checkpoint(config, checkpoint_path, "test", return_predictions=True)
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
    commit_hash = get_commit_hash()
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
            group=f'"commit_{commit_hash}"',
            name=f"{experiment_name}_commit_{commit_hash}",
            config=trial_config.model_dump(mode="json"),
        )
        save_json(run_context_path, current_run_context)
        save_json(output_dir / "resolved_config.json", trial_config.model_dump(mode="json"))
        logger = make_logger(output_dir / "train.log")

        expected_training_feature_signature = config_signature(feature_signature_payload(trial_config, "training"))
        expected_validation_feature_signature = config_signature(feature_signature_payload(trial_config, "validation"))
        expected_training_stats_signature = expected_training_feature_signature
        train_dataset = MosquitoFeatureDataset(
            feature_pickle_path=split_feature_path(trial_config, "training"),
            feature_stats_path=training_stats_path(trial_config),
            max_train_frames=max_train_frames(trial_config),
            training=True,
            normalize_features=trial_config.normalize_features,
            expected_feature_signature=expected_training_feature_signature,
            expected_stats_signature=expected_training_stats_signature,
        )
        val_dataset = MosquitoFeatureDataset(
            feature_pickle_path=split_feature_path(trial_config, "validation"),
            feature_stats_path=training_stats_path(trial_config),
            max_train_frames=None,
            training=False,
            normalize_features=trial_config.normalize_features,
            expected_feature_signature=expected_validation_feature_signature,
            expected_stats_signature=expected_training_stats_signature,
        )
        print(f"loading from {split_feature_path(trial_config, 'training')}")
        print(f"loading from {split_feature_path(trial_config, 'validation')}")
        if trial_config.normalize_features:
            print(f"loading from {training_stats_path(trial_config)}")

        train_loader = make_loader(train_dataset, trial_config.batch_size, True, trial_config.num_workers, device, pad_collate_fn)
        eval_batch_size = trial_config.eval_batch_size
        val_loader = make_loader(val_dataset, eval_batch_size, False, trial_config.num_workers, device, pad_collate_fn)

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
                "train_loss": round(train_metrics["loss"], 6),
                "train_species_loss": round(train_metrics["species_loss"], 6),
                "train_domain_loss": round(train_metrics["domain_loss"], 6),
                "train_species_accuracy": round(train_metrics["species_accuracy"], 6),
                "train_domain_accuracy": round(train_metrics["domain_accuracy"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_species_loss": round(val_metrics["species_loss"], 6),
                "val_domain_loss": round(val_metrics["domain_loss"], 6),
                "val_species_accuracy": round(val_metrics["species_accuracy"], 6),
                "val_species_balanced_accuracy": round(val_metrics["species_balanced_accuracy"], 6),
                "val_domain_accuracy": round(val_metrics["domain_accuracy"], 6),
                "val_domain_balanced_accuracy": round(val_metrics["domain_balanced_accuracy"], 6),
                "lr": optimizer.param_groups[0]["lr"],
            }
            for domain_name in DOMAIN_NAMES:
                val_species_ba = val_metrics.get(f"species_ba_{domain_name}")
                row[f"val_species_ba_{domain_name}"] = (
                    round(val_species_ba, 6) if val_species_ba is not None else None
                )

            append_metrics(output_dir / "metrics.csv", row)
            logger.info(row)
            wandb.log(row)

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
