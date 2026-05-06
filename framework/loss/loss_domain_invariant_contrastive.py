import torch

from framework.loss.loss_supervised_contrastive_positive_mask import loss_supervised_contrastive_positive_mask


def loss_domain_invariant_contrastive(
        embeddings: torch.Tensor, # (N, D)
        domain_labels: torch.Tensor, # (N,)
        temperature: float,
) -> torch.Tensor:
    # N -> number of samples in batch
    # D -> embedding size

    domain_labels = domain_labels.view(-1, 1) # (N, 1)
    positive_mask = torch.ne(domain_labels, domain_labels.T) # (N, N)

    return loss_supervised_contrastive_positive_mask(
        embeddings=embeddings,
        positive_mask=positive_mask,
        temperature=temperature,
    )
