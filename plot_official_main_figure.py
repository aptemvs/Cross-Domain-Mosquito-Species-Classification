"""Plot a paper-style per-species official evaluation figure.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the per-species official best-model figure for the paper.")
    parser.add_argument(
        "--input-json",
        type=str,
        default="technical_report_assets_current_split/per_species_official_best_model.json",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default="technical_report_assets_current_split/per_species_official_best_model_main_figure.png",
    )
    parser.add_argument(
        "--output-pdf",
        type=str,
        default="technical_report_assets_current_split/per_species_official_best_model_main_figure.pdf",
    )
    return parser.parse_args()


def load_rows(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot(rows, output_png: Path, output_pdf: Path) -> None:
    species = [row["species_short"] for row in rows]
    seen_means = [np.nan if row["BA_seen_mean"] is None else row["BA_seen_mean"] for row in rows]
    seen_stds = [0.0 if row["BA_seen_std"] is None else row["BA_seen_std"] for row in rows]
    unseen_means = [np.nan if row["BA_unseen_mean"] is None else row["BA_unseen_mean"] for row in rows]
    unseen_stds = [0.0 if row["BA_unseen_std"] is None else row["BA_unseen_std"] for row in rows]

    x = np.arange(len(species))
    width = 0.34

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(7.8, 3.6))

    seen_positions = x - width / 2
    unseen_positions = x + width / 2

    seen_mask = ~np.isnan(seen_means)
    unseen_mask = ~np.isnan(unseen_means)

    ax.bar(
        seen_positions[seen_mask],
        np.asarray(seen_means)[seen_mask],
        width=width,
        yerr=np.asarray(seen_stds)[seen_mask],
        capsize=3,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
        label="Seen",
    )
    ax.bar(
        unseen_positions[unseen_mask],
        np.asarray(unseen_means)[unseen_mask],
        width=width,
        yerr=np.asarray(unseen_stds)[unseen_mask],
        capsize=3,
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
        label="Unseen",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(species)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Species")
    ax.set_title("Per-species official evaluation (best model)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper right")

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    output_png = Path(args.output_png)
    output_pdf = Path(args.output_pdf)
    rows = load_rows(input_json)
    plot(rows, output_png, output_pdf)
    print(f"saved {output_png}")
    print(f"saved {output_pdf}")


if __name__ == "__main__":
    main()
