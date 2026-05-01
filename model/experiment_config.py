from pathlib import Path
from typing import Literal, Annotated

from pydantic import BaseModel, PositiveInt, PositiveFloat, Field, TypeAdapter

from const.model_backend import ModelBackend
from model.feature_extraction_config import FeatureExtractionConfig
from model.effecientat_config import EfficientATConfig


class _ExperimentBaseConfig(BaseModel):
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


class _MTRCNNExperimentConfig(_ExperimentBaseConfig):
    model_backend: Literal[ModelBackend.MTRCNN]


class _EfficientATExperimentConfig(_ExperimentBaseConfig):
    model_backend: Literal[ModelBackend.EFFICIENTAT]
    efficientat: EfficientATConfig


ExperimentConfig = TypeAdapter(
    Annotated[
        _MTRCNNExperimentConfig | _EfficientATExperimentConfig,
        Field(discriminator="model_backend"),
    ]
)
