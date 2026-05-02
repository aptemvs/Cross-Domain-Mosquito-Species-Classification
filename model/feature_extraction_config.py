from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    PositiveInt,
    PositiveFloat,
    Field,
    computed_field,
    model_validator,
    AfterValidator,
)

from validator.path import check_path_exists


class FeatureExtractionConfig(BaseModel):
    dataset_root: Annotated[Path, AfterValidator(check_path_exists)]
    train_ids_path: Annotated[Path, AfterValidator(check_path_exists)]
    val_ids_path: Annotated[Path, AfterValidator(check_path_exists)]
    test_ids_path: Annotated[Path, AfterValidator(check_path_exists)]

    sample_rate: PositiveInt
    normalize_waveform: bool
    normalize_features: bool
    hop_length_ms: PositiveFloat
    win_length_ms: PositiveFloat

    @computed_field
    @property
    def hop_length(self) -> int:
        return max(1, int(self.hop_length_ms * self.sample_rate / 1000))

    @computed_field
    @property
    def win_length(self) -> int:
        return max(1, int(self.win_length_ms * self.sample_rate / 1000))

    n_fft: PositiveInt = Field(default_factory=lambda data: data["win_length"])
    n_mels: PositiveInt
    f_min: PositiveInt = Field(default=0)
    f_max: PositiveInt = Field(default_factory=lambda data: data["sample_rate"] // 2)

    # ==========
    # Validators
    # ==========

    @model_validator(mode="after")
    def check_f_max(self):
        if self.f_max > self.sample_rate / 2:
            raise ValueError("f_max cannot exceed half of the sample rate")
        return self

    @model_validator(mode="after")
    def check_f_min(self):
        if self.f_min >= self.f_max:
            raise ValueError("f_min must be strictly less than f_max")
        return self

    @model_validator(mode="after")
    def check_n_fft(self):
        if self.n_fft < self.win_length:
            raise ValueError("n_fft must be less than win_length")
        return self
