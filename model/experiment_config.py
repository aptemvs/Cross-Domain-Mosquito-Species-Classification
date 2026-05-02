from pathlib import Path
from typing import Literal, Annotated

from pydantic import BaseModel, PositiveInt, PositiveFloat, Field, BeforeValidator

from const.model_backend import ModelBackend
from model.feature_extraction_config import FeatureExtractionConfig
from validator.to_list import to_list

type AutoList[T, *Args] = Annotated[list[T], BeforeValidator(to_list), *Args]


class _ExperimentBaseConfig(BaseModel):
    seed: AutoList[int]

    feature_root: Path
    output_root: Path

    feature_extraction: FeatureExtractionConfig

    normalize_features: bool

    batch_size: AutoList[PositiveInt]
    eval_batch_size: AutoList[PositiveInt]
    num_workers: PositiveInt
    train_crop_seconds: AutoList[PositiveFloat]
    epochs: PositiveInt
    early_stopping_min_epoch: PositiveInt
    early_stopping_patience: PositiveInt

    learning_rate: AutoList[PositiveFloat]
    weight_decay: AutoList[PositiveFloat]
    dropout: AutoList[PositiveFloat]

    device: str


class _MTRCNNExperimentConfig(_ExperimentBaseConfig):
    model_backend: Literal[ModelBackend.MTRCNN]


class _EfficientATExperimentConfig(_ExperimentBaseConfig):
    model_backend: Literal[ModelBackend.EFFICIENTAT]
    efficientat: EfficientATConfig


type ExperimentConfig = Annotated[
    _MTRCNNExperimentConfig | _EfficientATExperimentConfig,
    Field(discriminator="model_backend"),
]
