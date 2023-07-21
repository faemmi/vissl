# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import pprint
from typing import Union

import torch
import torch.distributed as dist
from classy_vision.generic.distributed_util import (
    all_reduce_sum,
    gather_from_all,
    get_rank,
    get_world_size,
)
from classy_vision.losses import ClassyLoss, register_loss
from torch import nn
import mlflow
import vissl.plotting as plotting
import vissl.utils.io as io
import vissl.utils.mantik as mantik
from vissl.config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.misc import get_indices_sparse


torch.cuda.empty_cache()


@register_loss("deepclusterv2_loss")
class DeepClusterV2Loss(ClassyLoss):
    """
    Loss used for DeepClusterV2 approach as provided in SwAV paper
    https://arxiv.org/abs/2006.09882

    Config params:
        DROP_LAST (bool): automatically inferred from DATA.TRAIN.DROP_LAST
        BATCHSIZE_PER_REPLICA (int): 256  # automatically inferred from
                                            DATA.TRAIN.BATCHSIZE_PER_REPLICA
        num_crops (int): 2                # automatically inferred from DATA.TRAIN.TRANSFORMS
        temperature (float): 0.1
        num_clusters (List[int]): [3000, 3000, 3000]
        kmeans_iters (int): 10
        crops_for_mb: [0]
        embedding_dim: 128
        num_train_samples (int): -1       # @auto-filled
    """

    def __init__(self, loss_config: AttrDict):
        super().__init__()

        self.loss_config = loss_config
        size_dataset = self.loss_config.num_train_samples

        size_memory_per_process = int(math.ceil(size_dataset * 1.0 / get_world_size()))

        if self.loss_config.DROP_LAST:
            size_memory_per_process -= (
                size_memory_per_process % self.loss_config.BATCHSIZE_PER_REPLICA
            )

        self.nmb_mbs = len(self.loss_config.memory_params.crops_for_mb)
        self.nmb_heads = len(self.loss_config.num_clusters)
        self.num_clusters = self.loss_config.num_clusters
        self.embedding_dim = self.loss_config.memory_params.embedding_dim
        self.crops_for_mb = self.loss_config.memory_params.crops_for_mb
        self.nmb_unique_idx = self.loss_config.BATCHSIZE_PER_REPLICA
        self.num_crops = self.loss_config.num_crops
        self.temperature = self.loss_config.temperature
        self.nmb_kmeans_iters = self.loss_config.kmeans_iters
        self.start_idx = 0

        local_rank, _ = get_machine_local_and_dist_rank()
        self.device = torch.device(
            "cpu" if mantik.cpu_usage_enabled() else f"cuda:{local_rank}"
        )

        self.register_buffer(
            "local_memory_embeddings",
            torch.zeros(self.nmb_mbs, size_memory_per_process, self.embedding_dim),
        )
        self.register_buffer(
            "local_memory_index", torch.zeros(size_memory_per_process).long()
        )
        self.register_buffer(
            "assignments", -1 * torch.ones(self.nmb_heads, size_dataset).long()
        )
        for i, k in enumerate(self.loss_config.num_clusters):
            self.register_buffer(
                "centroids" + str(i), torch.rand(k, self.embedding_dim)
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

        # Note (fabian.emmerich): Added extras
        self.register_buffer(
            "embeddings",
            torch.zeros(self.nmb_heads, self.nmb_mbs, size_dataset, self.embedding_dim),
        )
        self.register_buffer(
            "indexes", -1 * torch.ones(self.nmb_heads, size_dataset).long()
        )

        distance = torch.rand(self.nmb_heads, size_dataset)

        if mantik.cpu_usage_enabled():
            distance = distance.float()
        else:
            distance = distance.half()

        self.register_buffer("distance", -1 * distance)

        self.tensors_dir = f"{self.loss_config.output_dir}/tensors"
        self.plots_dir = f"{self.loss_config.output_dir}/plots"
        io.makedir(self.loss_config.output_dir)
        io.makedir(self.tensors_dir)
        io.makedir(self.plots_dir)

    @classmethod
    def from_config(cls, loss_config: AttrDict):
        """
        Instantiates DeepClusterV2Loss from configuration.

        Args:
            loss_config: configuration for the loss

        Returns:
            DeepClusterV2Loss instance.
        """
        return cls(loss_config)

    def forward(self, output: torch.Tensor, idx: int):
        logging.debug("Rank: %s, Forwarding for index %s", get_rank(), idx)
        output = nn.functional.normalize(output, dim=1, p=2)
        loss = 0
        for i in range(self.nmb_heads):
            scores = (
                torch.mm(output, getattr(self, "centroids" + str(i)).t())
                / self.temperature
            )
            loss += self.cross_entropy_loss(scores, self.assignments[i][idx])
        loss /= self.nmb_heads

        self.update_memory_bank(output, idx)
        return loss

    def init_memory(self, dataloader, model):
        logging.info(
            "Rank: %s, Start initializing memory banks for device %s",
            get_rank(),
            self.device.type,
        )
        start_idx = 0
        with torch.no_grad():
            for inputs in dataloader:
                nmb_unique_idx = len(inputs["data_idx"][0]) // self.num_crops
                index = inputs["data_idx"][0][:nmb_unique_idx].to(
                    device=self.device, non_blocking=True
                )
                # get embeddings
                outputs = []
                for crop_idx in self.crops_for_mb:
                    inp = inputs["data"][0][crop_idx].to(
                        device=self.device, non_blocking=True
                    )
                    outputs.append(nn.functional.normalize(model(inp)[0], dim=1, p=2))

                # fill the memory bank
                self.local_memory_index[start_idx : start_idx + nmb_unique_idx] = index
                for mb_idx, embeddings in enumerate(outputs):
                    self.local_memory_embeddings[mb_idx][
                        start_idx : start_idx + nmb_unique_idx
                    ] = embeddings
                start_idx += nmb_unique_idx
        logging.debug(
            "Rank: %s, Memory banks initialized, full first forward pass done",
            get_rank(),
        )

    def update_memory_bank(self, emb, idx):
        """Update memory banks.

        Parameters
        ----------
        embeddings : torch.Tensor, shape(n_crops, n_dim)
            Embeddings of each crop.
        idx : torch.Tensor, shape(n_crops * n_samples)
            Indexes of the data samples.
            E.g. if 2 crops and 2 samples given, ``idx`` has shape (4) and looks as
            ``tensor([1, 0, 1, 0])``, where 0 and 1 are the indexes of the two data samples.
        """
        logging.debug(
            "Rank: %s, Updating memory bank for index %s (start_index=%s)",
            get_rank(),
            idx,
            self.start_idx,
        )
        nmb_unique_idx = len(idx) // self.num_crops
        idx = idx[:nmb_unique_idx]
        self.local_memory_index[self.start_idx : self.start_idx + nmb_unique_idx] = idx
        for i, crop_idx in enumerate(self.crops_for_mb):
            self.local_memory_embeddings[i][
                self.start_idx : self.start_idx + nmb_unique_idx
            ] = emb[crop_idx * nmb_unique_idx : (crop_idx + 1) * nmb_unique_idx]
        self.start_idx += nmb_unique_idx
        logging.debug(
            "Rank: %s, Updated memory banks, full first forward pass done",
            get_rank(),
        )

    def cluster_memory(self):
        self.start_idx = 0

        # j defines which crops are used for the K-means run.
        # E.g. if the number of crops (``self.nmb_mbs``) is 2, and
        # ``self.num_clusters = [30, 30, 30, 30]``, the crops will
        # be used as following:
        #
        # 1. K=30, j=0
        # 2. K=30, j=1
        # 3. K=30, j=0
        # 4. K=30, j=1
        j = 0
        with torch.no_grad():
            for i_K, K in enumerate(self.num_clusters):
                # run distributed k-means

                centroids = torch.empty(K, self.embedding_dim).to(
                    device=self.device, non_blocking=True
                )
                if get_rank() == 0:
                    # Init centroids with elements from memory bank of rank 0 by
                    # taking K random samples from its local memory embeddings (i.e. the cropped
                    # samples) as centroids
                    random_idx = torch.randperm(len(self.local_memory_embeddings[j]))[
                        :K
                    ]

                    assert len(random_idx) >= K, "please reduce the number of centroids"
                    centroids = self.local_memory_embeddings[j][random_idx]

                # Send random centroids from rank 0 to all processes
                dist.broadcast(centroids, src=0)

                for n_iter in range(self.nmb_kmeans_iters + 1):
                    # E step
                    dot_products = torch.mm(
                        self.local_memory_embeddings[j], centroids.t()
                    )
                    distance, assignments = dot_products.max(dim=1)

                    # finish
                    if n_iter == self.nmb_kmeans_iters:
                        break
                    # M step
                    where_helper = get_indices_sparse(assignments.cpu().numpy())
                    counts = (
                        torch.zeros(K).to(device=self.device, non_blocking=True).int()
                    )
                    emb_sums = torch.zeros(K, self.embedding_dim).to(
                        device=self.device, non_blocking=True
                    )
                    for k in range(len(where_helper)):
                        if len(where_helper[k][0]) > 0:
                            emb_sums[k] = torch.sum(
                                self.local_memory_embeddings[j][where_helper[k][0]],
                                dim=0,
                            )
                            counts[k] = len(where_helper[k][0])

                    all_reduce_sum(
                        counts
                    )  # performing sum reduction of tensor over all processes
                    mask = counts > 0
                    all_reduce_sum(emb_sums)
                    centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                    # normalize centroids
                    centroids = nn.functional.normalize(centroids, dim=1, p=2)

                getattr(self, "centroids" + str(i_K)).copy_(centroids)

                # Gather results from all ranks
                assignments_all = gather_from_all(assignments)
                # To gather embeddings, make sure to gather using ``self.local_memory_embeddings[j]``
                embeddings_all = gather_from_all(self.local_memory_embeddings[j])
                indexes_all = gather_from_all(self.local_memory_index)
                distance_all = gather_from_all(distance)

                for tensor, value in [
                    (assignments_all, -1),
                    (embeddings_all, -1.0),
                    (indexes_all, -1),
                    (distance_all, -1.0),
                ]:
                    if _tensor_contains_value(tensor=tensor, value=value):
                        indexes = _tensor_contains_value_at(tensor=tensor, value=value)
                        logging.warning(
                            "After gathering from all ranks, tensor %s contains value %s at indexes %s",
                            tensor,
                            value,
                            indexes,
                        )

                self.assignments[i_K] = -1
                self.assignments[i_K][indexes_all] = assignments_all
                self.indexes[i_K] = -1
                self.indexes[i_K][indexes_all] = indexes_all
                self.distance[i_K] = -1.0
                self.distance[i_K][indexes_all] = distance_all
                # For the embeddings, make sure to use j for indexing
                self.embeddings[i_K][j] = -1.0
                self.embeddings[i_K][j][indexes_all] = embeddings_all

                j_prev = j
                j = (j + 1) % self.nmb_mbs

            epoch = mantik.get_current_epoch()
            epoch_comp = epoch + 1

            if (
                epoch_comp == 1
                or (epoch_comp <= 100 and epoch_comp % 25 == 0)
                or epoch_comp % 100 == 0
            ):
                logging.info(
                    "Saving clustering data on rank %s at epoch %s", get_rank(), epoch
                )

                centroids_last_iter = getattr(
                    self, f"centroids{len(self.num_clusters) - 1}"
                )
                torch.save(
                    centroids_last_iter, self._create_path("centroids.pt", epoch=epoch)
                )
                torch.save(
                    self.assignments, self._create_path("assignments.pt", epoch=epoch)
                )
                torch.save(
                    self.embeddings, self._create_path("embeddings.pt", epoch=epoch)
                )
                torch.save(self.indexes, self._create_path("indexes.pt", epoch=epoch))
                torch.save(
                    self.distance, self._create_path("distances.pt", epoch=epoch)
                )

                if get_rank() == 0:
                    # Save which random samples were used as the centroids.
                    torch.save(
                        random_idx,
                        self._create_path("centroid-indexes.pt", epoch=epoch),
                    )
                    plotting.deepclusterv2.embeddings.plot_embeddings_using_tsne(
                        embeddings=self.embeddings[-1],
                        # Use previous j since this represents which crops
                        # were used for last cluster iteration.
                        j=j_prev,
                        assignments=self.assignments[-1],
                        centroids=random_idx,
                        name=f"epoch-{epoch}-embeddings",
                        output_dir=self.plots_dir,
                    )
                    plotting.deepclusterv2.assignments.plot_abundance(
                        assignments=self.assignments[-1],
                        name=f"epoch-{epoch}-assignments-abundance",
                        output_dir=self.plots_dir,
                    )
                    plotting.deepclusterv2.assignments.plot_appearance_per_week(
                        assignments=self.assignments[-1],
                        name=f"epoch-{epoch}-appearance-per-week",
                        output_dir=self.plots_dir,
                    )

                    if mantik.tracking_enabled():
                        n_unassigned_samples = _calculate_number_of_unassigned_samples(
                            self.assignments[-1],
                        )
                        mantik.call_mlflow_method(
                            mlflow.log_metric,
                            "unassigned_samples",
                            n_unassigned_samples,
                        )
                        percent_unassigned_samples = (
                            n_unassigned_samples / self.assignments.shape[-1]
                        )
                        mantik.call_mlflow_method(
                            mlflow.log_metric,
                            "unassigned_samples_percent",
                            percent_unassigned_samples,
                        )

        logging.info(f"Rank: {get_rank()}, clustering of the memory bank done")

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def _create_path(self, file_name: str, epoch: int) -> str:
        return f"{self.tensors_dir}/epoch-{epoch}-{file_name}"


def _calculate_number_of_unassigned_samples(assignments: torch.Tensor) -> int:
    unassigned_sample_indexes = (assignments == -1).nonzero(as_tuple=True)[0]
    n_unassigned_samples = len(unassigned_sample_indexes)
    return n_unassigned_samples


def _tensor_contains_value(tensor: torch.Tensor, value: Union[int, float]) -> bool:
    indexes = _tensor_contains_value_at(tensor=tensor, value=value)
    [size] = indexes.size()
    return size != 0


def _tensor_contains_value_at(
    tensor: torch.Tensor, value: Union[int, float]
) -> torch.Tensor:
    return (tensor == value).nonzero(as_tuple=True)[0].int()
