from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ModelOutput:
    embeddings: torch.Tensor
    species_logits: torch.Tensor
    domain_logits: torch.Tensor
