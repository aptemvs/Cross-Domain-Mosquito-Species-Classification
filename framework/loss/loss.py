from dataclasses import dataclass
from typing import Self

import torch

from framework.loss.loss_domain_invariant_contrastive import loss_domain_invariant_contrastive
from framework.loss.loss_species_cohesion_contrastive import loss_species_cohesion_contrastive
from framework.loss.loss_species_conditional_distribution_alignment import \
    loss_species_conditional_distribution_alignment
from framework.model_output import ModelOutput
import torch.nn.functional as F


@dataclass(slots=True)
class LossItems:
    ScL: float
    DcL: float
    ScoL: float
    DicL: float
    SdaL: float
    total: float

    @staticmethod
    def zero():
        return LossItems(
            ScL=0.0,
            DcL=0.0,
            ScoL=0.0,
            DicL=0.0,
            SdaL=0.0,
            total=0.0,
        )

    def add_scaled(self, other: Self, batch_size: float) -> Self:
        self.ScL += other.ScL * batch_size
        self.DcL += other.DcL * batch_size
        self.ScoL += other.ScoL * batch_size
        self.DicL += other.DicL * batch_size
        self.SdaL += other.SdaL * batch_size
        self.total += other.total * batch_size

def loss_all(
        model_output: ModelOutput,
        species_labels: torch.Tensor,
        domain_labels: torch.Tensor,
) -> tuple[torch.Tensor, LossItems]:
    # see https://arxiv.org/abs/2510.00346v1 for source of losses
    species_logits = model_output.species_logits
    domain_logits = model_output.domain_logits
    embeddings = model_output.embeddings

    # todo get from config
    w_ScL = 1 # always 1 in paper -> other weights can change relative to it.
    w_DcL = 0.1
    w_ScoL = 0.1
    w_DicL = 0.1
    w_SdaL = 0.1

    ScoL_temperature = 0.01 # default from paper
    DicL_temperature = 0.01 # default from paper

    ScL = (
        F.cross_entropy(species_logits, species_labels) * w_ScL
        if w_ScL != 0
        else species_logits.new_tensor(0.0)
    )
    DcL = (
        F.cross_entropy(domain_logits, domain_labels) * w_DcL
        if w_DcL != 0
        else domain_logits.new_tensor(0.0)
    )
    ScoL = (
        loss_species_cohesion_contrastive(embeddings, species_labels, temperature=ScoL_temperature) * w_ScoL
        if w_ScoL != 0
        else embeddings.new_tensor(0.0)
    )
    DicL = (
            loss_domain_invariant_contrastive(embeddings, domain_labels, temperature=DicL_temperature) * w_DicL
            if w_DicL != 0
            else embeddings.new_tensor(0.0)
    )
    SdaL = (
        loss_species_conditional_distribution_alignment(embeddings, species_labels, domain_labels) * w_SdaL
        if w_SdaL != 0
        else embeddings.new_tensor(0.0)
    )

    total_loss = ScL + DcL + ScoL + DicL + SdaL

    return total_loss, LossItems(
        ScL=ScL.item(),
        DcL=DcL.item(),
        ScoL=ScoL.item(),
        DicL=DicL.item(),
        SdaL=SdaL.item(),
        total=total_loss.item(),
    )
