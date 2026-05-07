import torch
import torch.nn.functional as F

def loss_supervised_contrastive_positive_mask(
        embeddings: torch.Tensor, # (N, D)
        positive_mask: torch.Tensor, # (N, N)
        temperature: float,
) -> torch.Tensor:
    # N -> number of samples in batch
    # D -> embedding size

    device = embeddings.device
    batch_size = embeddings.size(0)

    # cos-sim matrix between embeddings
    embeddings = F.normalize(embeddings, dim=1) # (N, D)
    similarity = torch.matmul(embeddings, embeddings.T) / temperature # (N, N)

    # self masks -> similarity of diagonal ignored in loss
    self_mask = torch.eye(batch_size, device=device).bool() # (N, N)

    # positive mask (remove diagonal)
    positive_mask = positive_mask & (~self_mask) # (N, N)

    # softmax — exclude self from denominator (A(i) = {1..N}\{i} per paper Eq. 3)
    logits = similarity.masked_fill(self_mask, float('-inf')) # (N, N)
    log_prob = F.log_softmax(logits, dim=1) # (N, N)
    log_prob = log_prob.masked_fill(~positive_mask, 0.0)

    # (manually implemented, since we ignore self-similarity - but should be equivalent to above)
    # # - logits < 0 -> exponent stable
    # logits = similarity - similarity.max(dim=1, keepdim=True).values.detach() # (N, N)
    # exp_logits = torch.exp(logits) * no_self_mask.float() # (N, N)
    # log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12) # (N, N)

    # mean log prop - masked via count > 0, bc needs to be 0 otherwise
    positive_count = positive_mask.sum(dim=1) # (N,)
    positive_count_mask = positive_count > 0 # (N,)

    losses = (
            log_prob.sum(dim=1)
            / positive_count.clamp(min=1) # -> division by zero removed. quotient zero anyway
    ) # (N,)

    if positive_count_mask.any():
        return -losses[positive_count_mask].mean()
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)
