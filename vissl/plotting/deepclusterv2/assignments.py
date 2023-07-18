import pathlib
from typing import Tuple

import matplotlib as mpl
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


def plot_appearance_per_week(
    assignments: torch.Tensor, name: str, output_dir: str
) -> None:
    fig, ax = plt.subplots()

    reshaped = _create_assignments_subset(assignments)

    colors = _colors.create_colors_for_assigments(assignments)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "Custom cmap", colors, len(colors)
    )

    pcolormesh = ax.pcolormesh(reshaped, cmap=cmap)
    fig.colorbar(pcolormesh, ax=ax, label="Label")

    ax.set_title("Cluster appearance per day of week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Week")

    plt.savefig(pathlib.Path(output_dir) / f"{name}.pdf")


def _create_assignments_subset(assignments: torch.Tensor) -> torch.Tensor:
    largest_dividable_by_7 = _find_largest_dividable_by_7(assignments)
    # Cut after 365 days to limit plot to first year in data.
    subset = assignments[:largest_dividable_by_7]

    # Set -100 labels to -1 if requried to limit range
    # of colorbar.
    subset[(subset == -100).nonzero(as_tuple=True)[0]] = -1

    reshaped = _reshape_assignments_to_weeks(subset)

    return reshaped


def _find_largest_dividable_by_7(assignments: torch.Tensor) -> int:
    [size] = assignments.size()

    # Cut off at 364 in case of more than 364 labels, corresponding to 1 year
    if size > 364:
        return 364
    return int(size / 7) * 7


def _reshape_assignments_to_weeks(assignments: torch.Tensor) -> torch.Tensor:
    # Reshape to 7 days in x-axis, and weeks in  y-axis.
    reshaped = assignments.reshape((-1, 7))
    return reshaped
