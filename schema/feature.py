from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    PositiveInt,
    NonNegativeInt,
    PositiveFloat,
    Field,
    computed_field,
    model_validator,
    AfterValidator,
)

from validator import check_path_exists
from const.enum import Split
from framework.utilization import compute_signature


class FeatureExtractionConfig(BaseModel):
    dataset_root: Annotated[Path, AfterValidator(check_path_exists)]
    train_ids_path: Annotated[Path, AfterValidator(check_path_exists)]
    val_ids_path: Annotated[Path, AfterValidator(check_path_exists)]
    test_ids_path: Annotated[Path, AfterValidator(check_path_exists)]

    sample_rate: PositiveInt
    normalize_waveform: bool
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
    f_min: NonNegativeInt = Field(default=0)
    f_max: PositiveInt = Field(default_factory=lambda data: data["sample_rate"] // 2)

    # ==========
    # Functions
    # ==========

    def get_split_ids_path(self, split: Split) -> Path:
        match split:
            case Split.TRAINING:
                return self.train_ids_path
            case Split.TEST:
                return self.test_ids_path
            case Split.VALIDATION:
                return self.val_ids_path

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


class FeatureExtractionConfigDump(BaseModel):
    """
    A signed dump of the feature extraction config used to extract features of the given split
    """

    sample_rate: PositiveInt
    normalize_waveform: bool

    hop_length_ms: PositiveFloat
    win_length_ms: PositiveFloat
    hop_length: PositiveInt
    win_length: PositiveInt

    n_fft: PositiveInt
    n_mels: PositiveInt
    f_min: NonNegativeInt
    f_max: PositiveInt

    ids_path: Path
    ids_sha256: str

    @computed_field
    @property
    def signature(self) -> str:
        dump = self.model_dump(
            mode="json", exclude={"ids_path"}, exclude_computed_fields=True
        )
        return compute_signature(dump)
