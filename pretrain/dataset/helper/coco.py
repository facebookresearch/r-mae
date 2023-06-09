# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
import random
from io import BytesIO

import torch
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image

from pretrain.utils.distributed import get_rank, get_world_size


class CocoDataset(VisionDataset):
    def __init__(
        self,
        root,
        annFile,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        cache_first=False,
    ):
        super(CocoDataset, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.coco_api = COCO(annFile)
        self.ids = list(sorted(self.coco_api.imgs.keys()))
        self.cache_mode = cache_mode
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.ids) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        if cache_mode:
            self.cache = {}
            if cache_first:
                self.cache_images()

    def cache_images(self):
        indices = torch.arange(len(self.ids)).tolist()
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        offset = self.num_samples * self.rank
        indices = set(indices[offset : offset + self.num_samples])

        self.cache = {}
        for index, img_id in enumerate(self.ids):
            if index not in indices:
                continue

            path = self.coco_api.loadImgs(img_id)[0]["file_name"]
            with open(os.path.join(self.root, path), "rb") as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache:
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()

            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.ids)


class CocoDetection(CocoDataset):
    def __init__(
        self,
        root,
        annFile,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        cache_first=False,
    ):
        super(CocoDetection, self).__init__(
            root,
            annFile,
            num_replicas=num_replicas,
            rank=rank,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_mode=cache_mode,
            cache_first=cache_first,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned ``coco.loadAnns``,
        """
        coco_api = self.coco_api
        img_id = self.ids[index]
        ann_ids = coco_api.getAnnIds(imgIds=img_id)
        target = coco_api.loadAnns(ann_ids)

        path = coco_api.loadImgs(img_id)[0]["file_name"]

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CocoMask(CocoDataset):
    def __init__(
        self,
        root,
        maskFolder,
        annFile,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        cache_first=False,
        mask_type=None,
    ):
        super(CocoMask, self).__init__(
            root,
            annFile,
            num_replicas=num_replicas,
            rank=rank,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_mode=cache_mode,
            cache_first=cache_first,
        )
        assert mask_type in ("fh", None)
        self.mask_type = mask_type
        self.maskFolder = maskFolder

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, masks).
        """
        coco_api = self.coco_api
        img_id = self.ids[index]
        path = coco_api.loadImgs(img_id)[0]["file_name"]
        img = self.get_image(path)
        if self.mask_type == "fh":
            mask_path = path.replace("jpg", "npy")
            masks = np.load(os.path.join(self.maskFolder, mask_path))
            masks = torch.from_numpy(masks)
        else:
            masks = None

        return img, masks


class CocoCaption(CocoDataset):
    def __init__(
        self,
        root,
        annFile,
        num_replicas=None,
        rank=None,
        transform=None,
        target_transform=None,
        transforms=None,
        cache_mode=False,
        cache_first=False,
    ):
        super(CocoDetection, self).__init__(
            root,
            annFile,
            num_replicas=num_replicas,
            rank=rank,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
            cache_mode=cache_mode,
            cache_first=cache_first,
        )

        with open(annFile, "r") as f:
            self.coco = json.load(f)

        # sort 'images' field so that they are aligned with 'annotations'
        # i.e., in alphabetical order
        self.coco["images"] = sorted(self.coco["images"], key=lambda x: x["id"])
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco["images"], self.coco["annotations"]):
                assert img["file_name"][:-4] == ann["file_name"][:-4]
        assert tuple(map(lambda x: x["id"], self.coco["images"])) == tuple(self.ids)

    def __getitem__(self, index):
        ann_info = (
            self.coco["annotations"][index]
            if "annotations" in self.coco
            else self.coco["images"][index]
        )

        img_path = os.path.join(self.root, ann_info["file_name"]).replace(
            ".png", ".jpg"
        )

        img = self.get_image(img_path)
        w, h = img.size

        target = {}
        target["image_id"] = torch.tensor(
            [ann_info["image_id"] if "image_id" in ann_info else ann_info["id"]]
        )

        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])

        if "captions" in ann_info:
            target["captions"] = ann_info["captions"]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
