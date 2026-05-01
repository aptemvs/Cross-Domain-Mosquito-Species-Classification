from pathlib import Path

from pydantic import BaseModel, PositiveInt, PositiveFloat

from const.model_backend import ModelBackend
from model.feature_extraction_config import FeatureExtractionConfig


class ExperimentConfig(BaseModel):
    seed: int

    feature_root: Path
    output_root: Path

    feature_extraction: FeatureExtractionConfig

    batch_size: PositiveInt
    eval_batch_size: PositiveInt
    num_workers: PositiveInt
    train_crop_seconds: PositiveFloat
    epochs: PositiveInt
    early_stopping_min_epoch: PositiveInt
    early_stopping_patience: PositiveInt

    learning_rate: PositiveFloat
    weight_decay: PositiveFloat
    dropout: PositiveFloat

    device: str

    model_backend: ModelBackend
