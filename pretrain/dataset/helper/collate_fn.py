# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import collections
import re

import torch

string_classes = str
np_str_obj_array_pattern = re.compile(r"[SaUO]")


default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def _collate_sample2d(sample):
    assert sample[0]["image"].ndim == 3

    image_size = sample[0].get("image_size", None)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if len(sample) == 1 and image_size is None:
        new_sample = {"image": sample[0]["image"][None], "mask": None}
    else:
        new_sample = {}

        if image_size is None:
            total_shape = (sample[i]["image"].shape for i in range(len(sample)))
            shape = (len(sample), *(max(elem) for elem in zip(*total_shape)))
        else:
            max_channels = max(sample[i]["image"].shape[0] for i in range(len(sample)))
            shape = (len(sample), max_channels, image_size[0], image_size[1])
        new_sample["image"] = sample[0]["image"].new_zeros(shape)
        b, h, w = shape[0], shape[2], shape[3]
        new_sample["mask"] = sample[0]["image"].new_ones(b, h, w).bool()

        for i, elem in enumerate(sample):
            c, h, w = elem["image"].shape
            new_sample["image"][i, :c, :h, :w].copy_(elem["image"])
            new_sample["mask"][i, :h, :w] = False

    for key in sample[0].keys():
        if key in ("image", "image_size", "mask"):
            continue

        if isinstance(sample[0][key], torch.Tensor):
            new_sample[key] = torch.stack([elem[key] for elem in sample], dim=0)

    return new_sample


def collate2d(batch, iter_per_update=1):
    batch = list(zip(*batch))

    if iter_per_update == 1:
        new_batch = [_collate_sample2d(batch[0]), batch[1]]
    elif iter_per_update > 1:
        sample = batch[0]
        target = batch[1]

        batch_size = len(sample)

        split_size = (batch_size + iter_per_update - 1) // iter_per_update
        num_split = (batch_size + split_size - 1) // split_size

        new_batch = [
            [
                _collate_sample2d(sample[i * split_size : (i + 1) * split_size]),
                target[i * split_size : (i + 1) * split_size],
            ]
            for i in range(num_split)
        ]
    else:
        raise ValueError("iter_per_update should be greater than or equal to 1")

    return new_batch
