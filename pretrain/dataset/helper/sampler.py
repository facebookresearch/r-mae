# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import itertools
from collections import defaultdict
from typing import TypeVar, Optional, Iterator

import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset

from pretrain.utils.distributed import shared_random_seed


T_co = TypeVar("T_co", covariant=True)


SAMPLER_REGISTRY = {}


def register_sampler(name):
    def register_sampler_cls(cls):
        if name in SAMPLER_REGISTRY:
            raise ValueError("Cannot register duplicate sampler ({})".format(name))

        SAMPLER_REGISTRY[name] = cls
        return cls

    return register_sampler_cls


def build_sampler(config, dataset, other_args):
    training = config.training

    task_name = config.task
    dataset_name = config.dataset

    dataset_config_name = dataset_name + "_" + task_name
    dataset_config = config.dataset_config[dataset_config_name]
    sampler_type = dataset_config.get("sampler", "standard")

    if sampler_type == "standard":
        sampler = DistributedSampler(dataset, shuffle=other_args["shuffle"])
    elif sampler_type == "shard":
        sampler = ShardDistribtedSampler(dataset, shuffle=other_args["shuffle"])
    elif sampler_type == "infinite":
        batch_size = training.batch_size
        max_update = training.max_update
        max_epoch = training.max_epoch

        if max_epoch is not None:
            total_size = max_epoch * len(dataset)
        else:
            total_size = max_update * batch_size

        sampler = InfiniteDistributedSampler(
            dataset, total_size, shuffle=other_args["shuffle"]
        )
    elif sampler_type == "repeat_factor":
        batch_size = training.batch_size
        max_update = training.max_update
        max_epoch = training.max_epoch

        if max_epoch is not None:
            total_size = max_epoch * len(dataset)
        else:
            total_size = max_update * batch_size

        repeat_factors = (
            RepeatFactorDistributedSampler.repeat_factors_from_category_frequency(
                dataset.data.dataset_dicts,
                0.001,
            )
        )
        sampler = RepeatFactorDistributedSampler(
            repeat_factors, dataset, total_size, shuffle=other_args["shuffle"]
        )

    return sampler


class DistributedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RepeatFactorDistributedSampler(Sampler[T_co]):
    def __init__(
        self,
        repeat_factors: torch.Tensor,
        dataset: Dataset,
        total_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.total_size = total_size
        self.rank = rank
        self.shuffle = shuffle
        self.num_samples = math.ceil(total_size / self.num_replicas)
        if seed is None:
            seed = shared_random_seed()
        self.seed = int(seed)
        self.int_part = torch.trunc(repeat_factors)
        self.frac_part = repeat_factors - self.int_part

    @staticmethod
    def repeat_factors_from_category_frequency(dataset_dicts, repeat_thresh):
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids}, default=1.0)
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        start = self.rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self.num_replicas
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.
        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.
        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self.frac_part), generator=generator)
        rep_factors = self.int_part + (rands < self.frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self.shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm].tolist()
            else:
                yield from indices.tolist()


class InfiniteDistributedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        total_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.total_size = total_size
        self.rank = rank
        self.shuffle = shuffle
        self.num_samples = math.ceil(self.total_size / self.num_replicas)
        if seed is None:
            seed = shared_random_seed()
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[T_co]:
        start = self.rank
        yield from itertools.islice(
            self._infinite_indices(), start, None, self.num_replicas
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(len(self.dataset), generator=g).tolist()
            else:
                yield from torch.arange(len(self.dataset)).tolist()


class ShardDistribtedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()

        # indices += indices[: (self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            shuffled_idx = torch.randperm(len(indices), generator=g)
            indices = torch.tensor(indices)[shuffled_idx].tolist()

        if len(indices) < self.num_samples:
            indices += [indices[0]]

        if len(indices) > self.num_samples:
            indices = indices[:-1]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
