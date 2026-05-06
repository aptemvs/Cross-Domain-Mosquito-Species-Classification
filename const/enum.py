from enum import Enum


class ModelBackend(str, Enum):
    EFFICIENTAT = "EfficientAT"
    MTRCNN = "MTRCNN"

class Split(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
