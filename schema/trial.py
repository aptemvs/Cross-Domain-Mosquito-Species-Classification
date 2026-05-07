from pydantic import PositiveInt, PositiveFloat

from schema.experiment import ExperimentConfig


# Override AutoList properties
class TrialConfig(ExperimentConfig):
    seed: int  # pyright: ignore[reportIncompatibleVariableOverride]

    batch_size: PositiveInt  # pyright: ignore[reportIncompatibleVariableOverride]
    eval_batch_size: PositiveInt  # pyright: ignore[reportIncompatibleVariableOverride]
    train_crop_seconds: PositiveFloat  # pyright: ignore[reportIncompatibleVariableOverride]

    learning_rate: PositiveFloat  # pyright: ignore[reportIncompatibleVariableOverride]
    weight_decay: PositiveFloat  # pyright: ignore[reportIncompatibleVariableOverride]
    dropout: PositiveFloat  # pyright: ignore[reportIncompatibleVariableOverride]
