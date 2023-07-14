# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math
import pprint
import os

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
from vissl.config import AttrDict
from vissl.utils.env import get_machine_local_and_dist_rank
from vissl.utils.misc import get_indices_sparse
import vissl.utils.mantik as mantik

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
        self.device = torch.device("cpu" if mantik.cpu_usage_enabled() else f"cuda:{local_rank}")

        self.register_buffer(
            "local_memory_embeddings",
            torch.zeros(self.nmb_mbs, size_memory_per_process, self.embedding_dim),
        )
        self.register_buffer(
            "local_memory_index", torch.zeros(size_memory_per_process).long()
        )
        self.register_buffer(
            "assignments", -100 * torch.ones(self.nmb_heads, size_dataset).long()
        )
        for i, k in enumerate(self.loss_config.num_clusters):
            self.register_buffer(
                "centroids" + str(i), torch.rand(k, self.embedding_dim)
            )

        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-100)

        # Note (fabian.emmerich): Added extras
        self.register_buffer(
            "embeddings",
            torch.zeros(self.nmb_heads, self.nmb_mbs, size_dataset, self.embedding_dim),
        )
        self.register_buffer(
            "indexes", -100 * torch.ones(self.nmb_heads, size_dataset).long()
        )

        distance = torch.rand(self.nmb_heads, size_dataset)

        if mantik.cpu_usage_enabled():
            distance = distance.float()
        else:
            distance = distance.half()

        self.register_buffer("distance", -100 * distance)



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
        logging.info("Rank: %s, Start initializing memory banks for device %s", get_rank(), self.device.type)
        start_idx = 0
        with torch.no_grad():

            for inputs in dataloader:
                nmb_unique_idx = len(inputs["data_idx"][0]) // self.num_crops
                index = inputs["data_idx"][0][:nmb_unique_idx].to(device=self.device, non_blocking=True)
                # get embeddings
                outputs = []
                for crop_idx in self.crops_for_mb:
                    inp = inputs["data"][0][crop_idx].to(device=self.device, non_blocking=True)
                    outputs.append(nn.functional.normalize(model(inp)[0], dim=1, p=2))

                # fill the memory bank
                self.local_memory_index[start_idx : start_idx + nmb_unique_idx] = index
                for mb_idx, embeddings in enumerate(outputs):
                    self.local_memory_embeddings[mb_idx][
                        start_idx : start_idx + nmb_unique_idx
                    ] = embeddings
                start_idx += nmb_unique_idx
        logging.info(
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
        logging.debug("Rank: %s, Updating memory bank for index %s (start_index=%s)", get_rank(), idx, self.start_idx)
        nmb_unique_idx = len(idx) // self.num_crops
        idx = idx[:nmb_unique_idx]
        self.local_memory_index[self.start_idx : self.start_idx + nmb_unique_idx] = idx
        for i, crop_idx in enumerate(self.crops_for_mb):
            self.local_memory_embeddings[i][
                self.start_idx : self.start_idx + nmb_unique_idx
            ] = emb[crop_idx * nmb_unique_idx : (crop_idx + 1) * nmb_unique_idx]
        self.start_idx += nmb_unique_idx
        logging.info(
            "Rank: %s, Updated memory banks, full first forward pass done",
            get_rank(),
        )

    def cluster_memory(self):
        self.start_idx = 0
        j = 0
        with torch.no_grad():
            for i_K, K in enumerate(self.num_clusters):
                # run distributed k-means

                # init centroids with elements from memory bank of rank 0
                centroids = torch.empty(K, self.embedding_dim).to(device=self.device, non_blocking=True)
                if get_rank() == 0:
                    random_idx = torch.randperm(len(self.local_memory_embeddings[j]))[
                        :K
                    ]

                    assert len(random_idx) >= K, "please reduce the number of centroids"
                    centroids = self.local_memory_embeddings[j][random_idx]
                dist.broadcast(centroids, 0)

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
                    counts = torch.zeros(K).to(device=self.device, non_blocking=True).int()
                    emb_sums = torch.zeros(K, self.embedding_dim).to(
                        device=self.device,
                        non_blocking=True
                    )
                    for k in range(len(where_helper)):
                        if len(where_helper[k][0]) > 0:
                            emb_sums[k] = torch.sum(
                                self.local_memory_embeddings[j][where_helper[k][0]],
                                dim=0,
                            )
                            counts[k] = len(where_helper[k][0])


                    all_reduce_sum(counts) #performing sum reduction of tensor over all processes
                    mask = counts > 0
                    all_reduce_sum(emb_sums)
                    centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

                    # normalize centroids
                    centroids = nn.functional.normalize(centroids, dim=1, p=2)

                getattr(self, "centroids" + str(i_K)).copy_(centroids)

                # gather the assignments
                assignments_all = gather_from_all(assignments)
                embeddings_all = gather_from_all(self.local_memory_embeddings)
                indexes_all = gather_from_all(self.local_memory_index)
                distance_all = gather_from_all(distance)

                self.assignments[i_K] = -100
                self.assignments[i_K][indexes_all] = assignments_all

                self.embeddings[i_K] = -100
                for i in range(self.nmb_mbs):
                    self.embeddings[i_K][i][indexes_all] = embeddings_all[i]

                self.indexes[i_K] = -100
                self.indexes[i_K][indexes_all] = indexes_all

                self.distance[i_K] = -100
                self.distance[i_K][indexes_all] = distance_all

                j = (j + 1) % self.nmb_mbs

            epoch = mantik.get_current_epoch()
            epoch_comp = epoch + 1

            if epoch_comp == 1 or (epoch_comp <= 100 and epoch_comp % 25 == 0) or epoch_comp % 100 == 0:
                logging.info("Saving clustering data on rank %s at epoch %s", get_rank(), epoch)

                centroids_last_iter = getattr(self, f"centroids{len(self.num_clusters) - 1}")
                torch.save(centroids_last_iter, self._create_path("centroids.pt", epoch=epoch))
                torch.save(self.assignments, self._create_path("assignments.pt", epoch=epoch))
                torch.save(self.embeddings, self._create_path("embeddings.pt", epoch=epoch))
                torch.save(self.indexes, self._create_path("indexes.pt", epoch=epoch))
                torch.save(self.distance, self._create_path("distances.pt", epoch=epoch))

        logging.info(f"Rank: {get_rank()}, clustering of the memory bank done")

    def __repr__(self):
        repr_dict = {"name": self._get_name()}
        return pprint.pformat(repr_dict, indent=2)

    def _create_path(self, file_name: str, epoch: int) -> str:
        return f"{self.loss_config.output_dir}/epoch-{epoch}-{file_name}"
