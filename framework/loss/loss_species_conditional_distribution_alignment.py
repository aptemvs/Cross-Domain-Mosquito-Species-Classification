import torch
import torch.nn.functional as F


def loss_species_conditional_distribution_alignment(
    embeddings,        # (N, D)
    species_labels,    # (N,)
    domain_labels,     # (N,)
    sigma=1.0,
):
    embeddings = F.normalize(embeddings, dim=1)

    # full kernel
    dist_sq = torch.cdist(embeddings, embeddings, p=2).pow(2) # (N, N)
    k = torch.exp(-dist_sq / (2 * sigma**2))  # (N, N)

    total_loss = 0.0
    species_count = 0

    for species in species_labels.unique():
        # checking if loss makes sense for species
        sp_mask = (species_labels == species)
        idx = sp_mask.nonzero(as_tuple=True)[0]
        n_of_species = idx.numel()

        if n_of_species < 2:
            continue

        k_species = k[idx][:, idx]  # (N_species, N_species)
        domains_of_species = domain_labels[idx]

        unique_domains_of_species = domains_of_species.unique()
        d_of_species = unique_domains_of_species.numel()

        if d_of_species < 2:
            continue

        memberships = (domains_of_species[:, None] == unique_domains_of_species[None, :]).float() # (n_of_species, d_of_species)
        counts = memberships.sum(dim=0)  # (d_of_species,)

        # avoid divide-by-zero
        valid = counts > 0 # (d_of_species,)
        if valid.sum() < 2:
            continue

        # MMD calc:
        # sums of K over all group pairs
        K_sum = memberships.T @ k_species @ memberships  # (d_of_species, d_of_species)

        # normalize to expectations
        K_count = counts[:, None] * counts[None, :]  # (d_of_species, d_of_species)
        K_mean = K_sum / (K_count + 1e-8)  # (d_of_species, d_of_species)

        diag = torch.diag(K_mean) # (d_of_species,)
        MMD_sq = diag[:, None] + diag[None, :] - 2 * K_mean # (d_of_species, d_of_species)

        # take upper triangle (i < j)
        pair_mask = torch.triu(torch.ones_like(MMD_sq), diagonal=1).bool()
        pair_values = MMD_sq[pair_mask]

        if pair_values.numel() == 0:
            continue

        total_loss += pair_values.mean()
        species_count += 1

    if species_count == 0:
        return embeddings.sum() * 0.0

    return total_loss / species_count
