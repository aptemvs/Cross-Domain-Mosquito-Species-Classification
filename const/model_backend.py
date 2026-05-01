from enum import Enum


class ModelBackend(str, Enum):
    EFFICIENTAT = "EfficientAT"
    MTRCNN = "MTRCNN"
