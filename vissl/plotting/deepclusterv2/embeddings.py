import logging
import pathlib
import time
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import openTSNE
import torch
import vissl.plotting.deepclusterv2._colors as _colors


def plot_embeddings_using_tsne(
    embeddings: torch.Tensor, assignments: torch.Tensor, name: str, output_dir: str
) -> None:
    """Plot the embeddings of DCv2 using t-SNE.

    Args:
        embeddings (torch.Tensor, shape(n_crops, n_samples, n_embedding_dims)): Embeddings
            as produced by the ResNet.
        assignments (torch.Tensor, shape(n_samples)): Assignments by DCv2 for each
            sample.
            Used for coloring each sample in the plot.
        name (str): Name of the figure
        output_dir (str): Path where to save the figure.

    """
    start = time.time()
    logging.info("Creating plot for embeddings")

    for i in range(embeddings.shape[0]):
        _, ax = plt.subplots()

        ax.set_title(f"Embeddings for crops {i}")

        x, y = _fit_tsne(embeddings[i])
        colors = _colors.create_colors_for_assigments(assignments)

        ax = ax.scatter(x, y, c=colors, s=1)

        plt.savefig(pathlib.Path(output_dir) / f"{name}-crops-{i}.pdf")

        logging.info("Finished embeddings plot in %s seconds", time.time() - start)


def _fit_tsne(embeddings: torch.Tensor) -> Iterator[Tuple[float, float]]:
    start = time.time()
    logging.info("Fitting t-SNE")

    result = openTSNE.TSNE().fit(embeddings)

    logging.info("Finished fitting t-SNE in %s seconds", time.time() - start)

    return zip(*result)
