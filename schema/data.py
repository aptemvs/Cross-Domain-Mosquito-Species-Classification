from typing import ClassVar, Annotated
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PlainSerializer,
)
import numpy as np
import torch

from const.enum import Split
from schema.feature import FeatureExtractionConfigDump


class SplitMetadata(BaseModel):
    """
    Schema of the split metadata file
    """

    split: Split
    num_items: NonNegativeInt
    config: FeatureExtractionConfigDump


class SplitStatistics(BaseModel):
    """
    Schema of the split statistics file
    """

    num_frames: NonNegativeInt
    config: FeatureExtractionConfigDump
    mean: list[float]
    std: list[float]

class ExtractedFeature(BaseModel):
    """
    Schema of the extracted feature in the dataset
    """

    _MDS_COLUMNS: ClassVar[dict] = {
        "feature": "ndarray:float32",
        "file_id": "str",
        "num_frames": "int",
        "feature_dim": "int",
        "species": "str",
        "species_label": "int",
        "domain": "str",
        "domain_label": "int",
        "audio_path": "str",
    }
    model_config = ConfigDict(arbitrary_types_allowed=True)

    feature: np.ndarray
    file_id: str
    num_frames: NonNegativeInt
    feature_dim: NonNegativeInt
    species: str
    species_label: int
    domain: str
    domain_label: int
    audio_path: Annotated[Path, PlainSerializer(str, return_type=str)]


class DataLoaderBatch(BaseModel):
    """
    Schema of the batch item
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_id: list[str]
    features: torch.Tensor
    lengths: torch.Tensor
    species_labels: torch.Tensor
    domain_labels: torch.Tensor
    species: list[str]
    domain: list[str]
    audio_path: list[Path]
