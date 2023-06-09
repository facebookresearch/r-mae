# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import warnings

from torch.utils.data import DataLoader, DistributedSampler

from pretrain.dataset.helper import build_sampler
from pretrain.dataset.base import BaseDataset
from pretrain.utils.general import get_batch_size
from pretrain.utils.distributed import get_world_size


TASK_DATASET_REGISTRY = {}


__all__ = ["BaseDataset"]


def build_dataset(config, dataset_type, current_device):
    task_name = config.task
    dataset_name = config.dataset

    dataset_config_name = dataset_name + "_" + task_name
    dataset_config = config.dataset_config[dataset_config_name]

    if task_name not in TASK_DATASET_REGISTRY:
        raise ValueError("Task {} is not found.".format(task_name))
    elif dataset_name not in TASK_DATASET_REGISTRY[task_name]:
        raise ValueError(
            "Dataset {} is not found in task {}.".format(dataset_name, task_name)
        )

    if dataset_type not in dataset_config.imdb_files:
        warnings.warn(
            "Dataset type {} is not present in "
            "imdb_files of dataset config. Returning None. "
            "This dataset won't be used.".format(dataset_type)
        )
        return None

    imdb_file = dataset_config["imdb_files"][dataset_type]
    dataset = TASK_DATASET_REGISTRY[task_name][dataset_name](
        dataset_config,
        dataset_type,
        imdb_file,
        dataset_name=task_name,
        current_device=current_device,
        global_config=config,
    )

    dataset.init_processors()

    return dataset


def build_dataloader(config, dataset_type, dataset):
    training = config.training
    num_workers = training.num_workers
    pin_memory = training.pin_memory

    other_args = {}
    if dataset_type == "train":
        other_args["shuffle"] = True

        if get_world_size() > 1:
            other_args["sampler"] = build_sampler(config, dataset, other_args)
            other_args.pop("shuffle")
    else:
        other_args["shuffle"] = False

        if get_world_size() > 1:
            other_args["sampler"] = DistributedSampler(dataset, **other_args)
            other_args.pop("shuffle")

    if dataset_type == "train":
        other_args["batch_size"] = get_batch_size(training.batch_size)
    else:
        other_args["batch_size"] = 1

    if dataset.iter_per_update > 1:
        other_args["drop_last"] = True

    loader = DataLoader(
        dataset=dataset,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=dataset.get_collate_fn(),
        **other_args
    )

    if num_workers >= 0:
        # Suppress leaking semaphore warning
        os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"

    return loader, other_args.get("sampler", None)


def register_task(task_name, dataset_name):
    def register_dataset_cls(cls):
        if (
            task_name in TASK_DATASET_REGISTRY
            and dataset_name in TASK_DATASET_REGISTRY[task_name]
        ):
            raise ValueError(
                "Cannot register duplicate task ({}:{})".format(task_name, dataset_name)
            )
        elif not issubclass(cls, BaseDataset):
            raise ValueError(
                "Dataset ({}) must extend BaseDataset".format(cls.__name__)
            )

        if task_name in TASK_DATASET_REGISTRY:
            TASK_DATASET_REGISTRY[task_name][dataset_name] = cls
        else:
            TASK_DATASET_REGISTRY[task_name] = {dataset_name: cls}
        return cls

    return register_dataset_cls


def get_task_list():
    return tuple(TASK_DATASET_REGISTRY.keys())


datasets_dir = os.path.dirname(__file__)
for file in os.listdir(datasets_dir):
    path = os.path.join(datasets_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        dataset_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("pretrain.dataset." + dataset_name)
