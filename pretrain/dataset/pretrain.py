# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
from torch.utils.data import ConcatDataset

from pretrain.dataset import BaseDataset, register_task
from pretrain.dataset.helper import collate2d, ImageDataset, CocoMask


class VisionPretrain(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        if "name" in kwargs:
            dataset_name = kwargs["name"]
        elif "dataset_name" in kwargs:
            dataset_name = kwargs["dataset_name"]
        else:
            dataset_name = "pretrain"

        super().__init__(
            config,
            dataset_name,
            dataset_type,
            current_device=kwargs["current_device"],
            global_config=kwargs["global_config"],
        )

    def get_collate_fn(self):
        return partial(collate2d, iter_per_update=self.iter_per_update)

    def get_answer_size(self):
        return None


class PretrainWithMask(VisionPretrain):
    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        super().__init__(config, dataset_type, imdb_file, **kwargs)

        mask_type = config.get("mask_type", None)
        assert mask_type in ("fh", None)
        self.mask_type = mask_type
        self.cache_mode = config.get("sampler", "standard") == "shard"

        self.rmae_sampling = self._global_config.model == "rmae"
        if self.rmae_sampling:
            rmae_vit_arch = self._global_config.model_config["rmae"].mae_vit.arch

            if "base_patch16" in rmae_vit_arch:
                self.patch_size = 16
            elif "large_patch16" in rmae_vit_arch:
                self.patch_size = 16
            elif "huge_patch14" in rmae_vit_arch:
                self.patch_size = 14

            rmae_vit_params = self._global_config.model_config["rmae"].mae_vit.params
            self.num_region = rmae_vit_params.num_region
            self.region_mask_ratio = rmae_vit_params.region_mask_ratio
            self.region_sample_type = rmae_vit_params.region_sample_type
            self.filter_small_patches = config.get("filter_small_patches", False)

            assert self.region_sample_type in ("random", "random_fg")

    def _process_mask(self, sample, target):
        masks = target.pop("masks")

        assert (
            self.mask_type == "fh"
        ), f"FH masks should be used (mask_type={self.mask_type})"
        processed_masks = []
        for mask in masks.unbind(0):
            mask = torch.unique(mask).unsqueeze(-1).unsqueeze(-1) == mask
            processed_masks.append(mask)
        masks = torch.cat(processed_masks, dim=0)

        if self.rmae_sampling:
            if self.filter_small_patches:
                assert masks.shape[1] == masks.shape[2]
                ph = pw = masks.shape[1] // self.patch_size
                b = masks.shape[0]
                patches = masks.view(b, ph, self.patch_size, pw, self.patch_size)
                patches = patches.any(-1).any(2).view(b, ph * pw)
                keep = patches.sum(dim=1, dtype=torch.int32) >= 2
                masks = masks[keep]

            if self.region_sample_type == "random_fg":
                assert masks.shape[1] == masks.shape[2]
                h = w = masks.shape[1] // self.patch_size
                b = masks.shape[0]

                patches = masks.view(b, h, self.patch_size, w, self.patch_size)
                patches = patches.any(-1).any(2).view(b, h * w)

                len_keep = int((1 - self.region_mask_ratio) * h * w)
                shuffle_ids = torch.randperm(h * w)
                keep_ids = shuffle_ids[:len_keep]
                keep = patches[:, keep_ids].any(dim=1)
                masks = masks[keep]

                sample["shuffle_ids"] = shuffle_ids

            n, h, w = masks.shape
            if n == 0:
                masks = torch.zeros(self.num_region, h, w)
            elif n < self.num_region:
                repeat = (self.num_region + n - 1) // n
                ids = torch.tensor(list(range(n)) * repeat)
                ids = ids[torch.randperm(ids.shape[0])][: self.num_region]
                masks = masks[ids]
            else:
                ids = torch.randperm(n)[: self.num_region]
                masks = masks[ids]

        sample["region"] = masks.float()

        return sample, target


@register_task("pretrain", "coco")
class COCOPretrain(PretrainWithMask):
    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        super().__init__(config, dataset_type, imdb_file, **kwargs)

        coco_dataset = []
        for img_folder, mask_folder, anno_file in zip(
            imdb_file["image_folder"],
            imdb_file["mask_folder"],
            imdb_file["anno_file"],
        ):
            coco_dataset.append(
                CocoMask(
                    self._get_absolute_path(img_folder),
                    self._get_absolute_path(mask_folder),
                    self._get_absolute_path(anno_file),
                    cache_mode=self.cache_mode,
                    mask_type=self.mask_type,
                )
            )
        self.coco_dataset = ConcatDataset(coco_dataset)

    def __len__(self):
        return len(self.coco_dataset)

    def __getitem__(self, idx):
        sample = {}

        if self.mask_type == "fh":
            img, masks = self.coco_dataset[idx]
            target = {"image_id": idx, "masks": masks}
        else:
            img, masks = self.coco_dataset[idx]
            target = {"image_id": idx}

        sample["image"] = img
        if self._dataset_type == "train":
            sample, target = self.image_train_processor(sample, target)
        else:
            sample, target = self.image_test_processor(sample, target)

        if self.mask_type is not None:
            sample, target = self._process_mask(sample, target)

        return sample, target


@register_task("pretrain", "imnet")
class ImageNetPretrain(PretrainWithMask):
    def __init__(self, config, dataset_type, imdb_file, **kwargs):
        super().__init__(config, dataset_type, imdb_file, **kwargs)

        mask_folder = self._get_absolute_path(imdb_file["mask_folder"])
        self.imnet_dataset = ImageDataset(
            self._get_absolute_path(imdb_file["image_folder"]),
            root_mask=mask_folder,
            mask_type=self.mask_type,
            cache_mode=self.cache_mode,
            cache_mask=config.get("cache_mask", False),
        )

    def __len__(self):
        return len(self.imnet_dataset)

    def __getitem__(self, idx):
        sample = {}

        if self.mask_type == "fh":
            img, masks = self.imnet_dataset[idx]
            target = {"image_id": idx, "masks": masks}
        else:
            img, masks = self.imnet_dataset[idx]
            target = {"image_id": idx}

        sample["image"] = img
        if self._dataset_type == "train":
            sample, target = self.image_train_processor(sample, target)
        else:
            sample, target = self.image_test_processor(sample, target)

        if self.mask_type is not None:
            sample, target = self._process_mask(sample, target)

        return sample, target
