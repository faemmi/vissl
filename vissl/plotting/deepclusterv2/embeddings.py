import logging
import time

import matplotlib.pyplot as plt
import torch


def plot_embeddings_using_tsne(embeddings: torch.Tensor) -> plt.Figure:
    start = time.time()
    logging.info("Creating plot for embeddings")
    logging.info("Finished embeddings plot in %s seconds", time.time() - start)
