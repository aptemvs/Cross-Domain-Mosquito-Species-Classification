import sys
import importlib
from contextlib import chdir
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


class EfficientATClassifier(nn.Module):

    def __init__(self, config: dict, num_species_classes: int, num_domain_classes: int) -> None:
        super().__init__()

        self.backbone = _get_backbone(config)
        if bool(config["efficientat_freeze"]):
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False

        feature_dim = self._infer_efficientat_feature_dim(config)
        embedding_dim = int(config["efficientat_embedding_dim"])
        dropout = float(config["dropout"])

        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.activation = nn.GELU()
        self.embedding_dropout = nn.Dropout(dropout)
        self.species_classifier = nn.Linear(embedding_dim, num_species_classes)
        self.domain_classifier = nn.Linear(embedding_dim, num_domain_classes)

    def _infer_efficientat_feature_dim(self, config) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, int(config["n_mels"]), config["efficientat_input_dim_t"])
            _, features = self.backbone(dummy)

        feature_dim = int(features.shape[1])
        print(f'efficientat feature dim inferred as {feature_dim=!r}')

        return int(features.shape[1])

    def forward(self, features: torch.Tensor, _lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = _transpose_input(features)
        _, backbone_features = self.backbone(x)
        if backbone_features.dim() == 1:
            backbone_features = backbone_features.unsqueeze(0)

        embedding = self.activation(self.embedding(backbone_features))
        embedding = self.embedding_dropout(embedding)
        return {
            "species_logits": self.species_classifier(embedding),
            "domain_logits": self.domain_classifier(embedding),
        }


def _transpose_input(features: torch.Tensor) -> torch.Tensor:
    if features.dim() == 3:
        # EfficientAT expects [batch, 1, mels, frames].
        # transposes input from [batch, frames, mels] to expected one.
        return features.transpose(1, 2).unsqueeze(1)
    if features.dim() == 4 and features.size(1) == 1:
        return features
    raise ValueError(
        "Expected features shaped [batch, frames, mels] or [batch, 1, mels, frames], "
        f"but got {tuple(features.shape)}"
    )


def _efficientat_root() -> Path:
    return Path(__file__).resolve().parents[1] / "third_party" / "EfficientAT"


def _get_backbone(config) -> nn.Module:
    root = _efficientat_root()
    if not root.exists():
        raise RuntimeError("./third_party/EfficientAT folder doesnt exist."
                           "See README.md for how to checkout submodule.")

    # EfficientAT is imported from a git submodule not as a package.
    # so we add its repo root to sys.path to make its relative imports work.
    efficientat_root_str = str(root)
    if efficientat_root_str not in sys.path:
        sys.path.insert(0, efficientat_root_str)

    # The get_model call is performed with its root as context.
    # bc uses relative paths for a few folders e.g. weights cache.
    with chdir(root):
        mn_module = importlib.import_module("models.mn.model")
        get_mn = getattr(mn_module, "get_model")

        # get_mn currently prints model graph
        # (we could suppress this, and print our own if needed)
        return get_mn(
            # see /third_party/EfficientAT/README.md for correct params
            # for pretrained models. some have to match.
            pretrained_name=str(config["efficientat_pretrained_name"]),
            width_mult=float(config["efficientat_width_mult"]),
            input_dim_f=int(config["n_mels"]),
            input_dim_t=config["efficientat_input_dim_t"],
        )
