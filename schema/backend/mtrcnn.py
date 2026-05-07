from typing import Literal

from pydantic import BaseModel

from const.enum import ModelBackend


class MTRCNNParams(BaseModel):
    pass


class MTRCNNBackend(MTRCNNParams):
    model: Literal[ModelBackend.MTRCNN]
