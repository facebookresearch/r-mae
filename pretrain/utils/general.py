# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import collections
import re
import sys

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from pretrain.utils.distributed import get_world_size, synchronize


string_classes = str


def clip_grad_norm(params, max_norm, with_name=False):
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(params, max_norm)
    else:
        if with_name:
            device = params[0][1].grad.device
            for name, p in params:
                if p.grad is None:
                    print(name)
            synchronize()

            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(p.grad.detach(), 2.0).to(device) for _, p in params]
                ),
                2.0,
            )
        else:
            device = params[0].grad.device
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(p.grad.detach(), 2.0).to(device) for p in params]
                ),
                2.0,
            )
        return total_norm


def get_clones(module, N):
    if N == 0:
        return []
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm, out_channels):
    if norm == "LN":
        return nn.LayerNorm(out_channels)
    if norm == "BN":
        return nn.BatchNorm2d(out_channels)
    if norm == "GN":
        return nn.GroupNorm(32, out_channels)

    raise RuntimeError(f"norm layer should be BN | LN | GN, not {norm}")


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def filter_grads(parameters, with_name=False):
    if with_name:
        return [(name, param) for name, param in parameters if param.requires_grad]
    return [param for param in parameters if param.requires_grad]


def get_root():
    root_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.abspath(os.path.join(root_folder, ".."))

    return root_folder


def get_cache_dir(cache_dir):
    # If cache_dir path exists do not join to mmf root
    if not os.path.exists(cache_dir):
        cache_dir = os.path.join(get_root(), cache_dir)
    return cache_dir


def get_batch_size(batch_size):
    world_size = get_world_size()

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    return batch_size // world_size


def get_absolute_path(paths):
    # String check should be first as Sequence would pass for string too
    if isinstance(paths, str):
        if not os.path.isabs(paths):
            root_dir = get_root()
            paths = os.path.join(root_dir, paths)
        return paths
    elif isinstance(paths, collections.abc.Iterable):
        return [get_absolute_path(path) for path in paths]
    else:
        raise TypeError("Paths passed to dataset should either be " "string or list")


def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024))
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def print_model_parameters(model, writer, return_only=False):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        writer.write(
            "Total Parameters: {}. Trained Parameters: {}".format(
                total_params, trained_params
            )
        )
    return total_params, trained_params


def get_optimizer_parameters(model):
    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    )
    has_custom = (
        hasattr(model.module, "get_optimizer_parameters")
        if is_parallel
        else hasattr(model, "get_optimizer_parameters")
    )

    if has_custom:
        parameters = (
            model.module.get_optimizer_parameters()
            if is_parallel
            else model.get_optimizer_parameters()
        )
    else:
        parameters = filter_grads(model.parameters())

    return parameters


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    return torchvision.ops.misc.interpolate(
        input, size, scale_factor, mode, align_corners
    )


np_str_obj_array_pattern = re.compile(r"[SaUO]")


def data_to_tensor(data):
    data_type = type(data)

    if isinstance(data, torch.Tensor):
        return data
    elif (
        data_type.__module__ == "numpy"
        and data_type.__name__ != "str_"
        and data_type.__name__ != "string_"
    ):
        if data_type.__name__ == "ndarray" or data_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(data.dtype.str) is not None:
                return data

            return torch.as_tensor(data)
        elif data.shape == ():
            return torch.as_tensor([data.item()])

    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float32)
    elif isinstance(data, int):
        return torch.tensor([data])
    elif isinstance(data, string_classes):
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: data_to_tensor(value) for key, value in data.items()}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return data_type(*(data_to_tensor(elem) for elem in data))
    elif isinstance(data, collections.abc.Sequence):
        return [data_to_tensor(elem) for elem in data]


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, "w")


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
