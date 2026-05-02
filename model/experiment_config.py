from pathlib import Path
from typing import Literal, Annotated

from pydantic import BaseModel, PositiveInt, PositiveFloat, Field

from const.model_backend import ModelBackend
from model.feature_extraction_config import FeatureExtractionConfig
from model.effecientat_config import EfficientATConfig

type MaybeList[T] = T | list[T]

class _ExperimentBaseConfig(BaseModel):
    seed: MaybeList[int] = Field(union_mode='left_to_right')

    feature_root: Path
    output_root: Path

    feature_extraction: FeatureExtractionConfig

    normalize_features: bool

    batch_size: MaybeList[PositiveInt] = Field(union_mode='left_to_right')
    eval_batch_size: MaybeList[PositiveInt] = Field(union_mode='left_to_right')
    num_workers: PositiveInt
    train_crop_seconds: MaybeList[PositiveFloat] = Field(union_mode='left_to_right')
    epochs: PositiveInt
    early_stopping_min_epoch: PositiveInt
    early_stopping_patience: PositiveInt

    learning_rate: MaybeList[PositiveFloat] = Field(union_mode='left_to_right')
    weight_decay: MaybeList[PositiveFloat] = Field(union_mode='left_to_right')
    dropout: MaybeList[PositiveFloat] = Field(union_mode='left_to_right')

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
