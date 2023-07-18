import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import vissl.plotting.deepclusterv2._colors as _colors


def plot_abundance(assignments: torch.Tensor, name: str, output_dir: str) -> None:
    labels = np.arange(int(assignments.min()), int(assignments.max()) + 1, 1, dtype=int)
    x_lims = labels.min() - 0.5, labels.max() + 0.5
    bins = np.arange(x_lims[0], x_lims[1] + 1.0, 1.0)

    _plot_histogram(
        assignments=assignments,
        bins=bins,
        y_label="Total abundance",
        labels=labels,
        x_lims=x_lims,
        density=False,
        outfile=pathlib.Path(output_dir) / f"{name}-total.pdf",
    )
    _plot_histogram(
        assignments=assignments,
        bins=bins,
        y_label="Relative abundance",
        labels=labels,
        x_lims=x_lims,
        density=True,
        outfile=pathlib.Path(output_dir) / f"{name}-relative.pdf",
    )


def _plot_histogram(
    assignments: torch.Tensor,
    bins: np.ndarray,
    y_label: str,
    labels: np.ndarray,
    x_lims: Tuple[float, float],
    density: bool,
    outfile: str,
) -> Tuple[plt.Figure, plt.Axes]:
    _, ax = plt.subplots()

    _, _, patches = ax.hist(assignments, bins=bins, density=density)

    ax.set_title(f"{y_label.capitalize()} per cluster")

    ax.set_ylabel(y_label)
    ax.set_xlabel("Label")
    ax.set_xlim(*x_lims)
    ax.set_xticks(labels)
    ax.set_xticklabels(labels)

    colors = _colors.create_colors_for_labels(labels)

    if len(colors) != len(patches):
        raise RuntimeError(
            "Length of colors does not match number of patches in histogram"
        )

    for color, patch in zip(colors, patches):
        patch.set_facecolor(color)

    plt.savefig(outfile)
