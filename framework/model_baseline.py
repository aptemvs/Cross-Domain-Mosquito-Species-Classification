"""MTRCNN model definition for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        dilation: tuple[int, int],
        padding: tuple[int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.pool(x)
        return self.dropout(x)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        time_kernel = self.kernel_size[0]
        time_dilation = self.dilation[0]
        time_padding = self.padding[0]
        conv_lengths = lengths + 2 * time_padding - time_dilation * (time_kernel - 1)
        return torch.div(conv_lengths.clamp_min(0), 2, rounding_mode="floor")


class MTRCNNBranch(nn.Module):
    def __init__(
        self,
        stage_specs: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
        dropout: float,
        n_mels: int,
    ) -> None:
        super().__init__()
        channels = [1, 16, 32, 64]
        self.stages = nn.ModuleList(
            [
                ConvStage(
                    in_channels=channels[index],
                    out_channels=channels[index + 1],
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
                for index, (kernel_size, dilation, padding) in enumerate(stage_specs)
            ]
        )
        self.frequency_projection = nn.Linear(self._infer_frequency_bins(n_mels), 1)

    def _infer_frequency_bins(self, n_mels: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, 128, n_mels)
            for stage in self.stages:
                x = stage(x)
        return x.shape[-1]

    def forward(self, x: torch.Tensor, frame_lengths: torch.Tensor) -> torch.Tensor:
        lengths = frame_lengths
        for stage in self.stages:
            x = stage(x)
            lengths = stage.output_lengths(lengths)

        pooled = masked_mean_max(x, lengths)
        pooled = self.frequency_projection(pooled).squeeze(-1)
        return F.relu(pooled, inplace=True)


def masked_mean_max(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    valid_lengths = lengths.clamp(min=1, max=x.size(2))
    time_index = torch.arange(x.size(2), device=x.device).view(1, 1, -1, 1)
    mask = time_index < valid_lengths.view(-1, 1, 1, 1)

    masked_sum = (x * mask).sum(dim=2)
    masked_mean = masked_sum / valid_lengths.view(-1, 1, 1)

    masked_max = x.masked_fill(~mask, float("-inf")).max(dim=2).values
    masked_max = torch.where(torch.isfinite(masked_max), masked_max, torch.zeros_like(masked_max))
    return masked_mean + masked_max


class MTRCNNClassifier(nn.Module):
    def __init__(self, config, num_species_classes: int, num_domain_classes: int) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm2d(config["n_mels"])

        self.kernel_3_branch = MTRCNNBranch(
            stage_specs=[
                ((3, 3), (1, 1), (1, 1)),
                ((3, 3), (2, 1), (2, 0)),
                ((3, 3), (3, 1), (3, 0)),
            ],
            dropout=config["dropout"],
            n_mels=config["n_mels"],
        )
        self.kernel_5_branch = MTRCNNBranch(
            stage_specs=[
                ((5, 5), (1, 1), (2, 2)),
                ((5, 5), (2, 1), (4, 1)),
                ((5, 5), (3, 1), (6, 1)),
            ],
            dropout=config["dropout"],
            n_mels=config["n_mels"],
        )
        self.kernel_7_branch = MTRCNNBranch(
            stage_specs=[
                ((7, 7), (1, 1), (3, 3)),
                ((7, 7), (2, 1), (6, 2)),
                ((7, 7), (3, 1), (9, 2)),
            ],
            dropout=config["dropout"],
            n_mels=config["n_mels"],
        )

        self.embedding = nn.Linear(64 * 3, 32)
        self.species_classifier = nn.Linear(32, num_species_classes)
        self.domain_classifier = nn.Linear(32, num_domain_classes)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        x = features.unsqueeze(1).transpose(1, 3)
        x = self.input_bn(x)
        x = x.transpose(1, 3)

        branch_outputs = [
            self.kernel_3_branch(x, lengths),
            self.kernel_5_branch(x, lengths),
            self.kernel_7_branch(x, lengths),
        ]
        features = torch.cat(branch_outputs, dim=1)
        embedding = F.gelu(self.embedding(features))
        return {
            "species_logits": self.species_classifier(embedding),
            "domain_logits": self.domain_classifier(embedding),
        }
