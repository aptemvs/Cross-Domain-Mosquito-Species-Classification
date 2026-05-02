from pydantic import BaseModel, PositiveInt


class EfficientATConfig(BaseModel):
    pretrained_name: str
    width_mult: float
    input_dim_t: PositiveInt
    embedding_dim: PositiveInt
    freeze: bool
