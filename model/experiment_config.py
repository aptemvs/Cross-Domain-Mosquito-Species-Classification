from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, PositiveInt, PositiveFloat, Field, BeforeValidator

from model.backend import EfficientATBackend, MTRCNNBackend
from model.feature_extraction_config import FeatureExtractionConfig
from validator.to_list import to_list

type AutoList[T, *Args] = Annotated[list[T], BeforeValidator(to_list), *Args]


class ExperimentConfig(BaseModel):
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
    backend: Annotated[MTRCNNBackend | EfficientATBackend, Field(discriminator="model")]
