# BioDCASE 2026 Task5 - CD-MSC

Fork of the [CD-MSC baseline repository](https://github.com/Yuanbo2020/CD-MSC/) for the Biocase Task 5 2026.

Baseline readme in [README_BASELINE.md](./README_BASELINE.md).

## Wandb

To log to wandb, login within an active conda environment
with installed requirements:

```bash
wandb login
```

## Third-Party Dependencies

EfficientAT is tracked as a git submodule in `third_party/EfficientAT`.

### Clone with submodules
When checking out fresh:

```bash
git clone --recurse-submodules git@github.com:aptemvs/Cross-Domain-Mosquito-Species-Classification.git
```

### In an existing repo
Add submodules via:

```bash
git submodule update --init --recursive
```

### Pretrained weights

The pretrained weights for the EfficientAT models are loaded from their GitHub releases,
as implemented in the EfficientAT repository. Cached locally in `third_party/EfficientAT/resources/`.
