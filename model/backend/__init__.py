from typing import Literal

from pydantic import BaseModel

from const.model_backend import ModelBackend

from model.backend.effecientat import EfficientATConfig


class MTRCNNBackend(BaseModel):
    model: Literal[ModelBackend.MTRCNN]


class EfficientATBackend(EfficientATConfig):
    model: Literal[ModelBackend.EFFICIENTAT]
