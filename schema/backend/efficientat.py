from typing import Literal

from pydantic import BaseModel, PositiveInt

from const.enum import ModelBackend


class EfficientATParams(BaseModel):
    pretrained_name: str
    width_mult: float
    input_dim_t: PositiveInt
    embedding_dim: PositiveInt
    freeze: bool


class EfficientATBackend(EfficientATParams):
    model: Literal[ModelBackend.EFFICIENTAT]
