# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from io import BytesIO

import torch
import numpy as np
import torchvision
import cv2
from PIL import Image


class ImageDataset(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root: str,
        root_mask=None,
        transform=None,
        target_transform=None,
        is_valid_file=None,
        cache_mode=False,
        mask_type=None,
        reader_type="pil",
        cache_mask=False,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

        assert mask_type in ("fh", None)
        self.mask_type = mask_type

        self.root_mask = root_mask
        self.cache_mode = cache_mode
        self.cache_mask = cache_mask
        self.reader_type = reader_type
        if cache_mode:
            self.cache = {}
        if cache_mode or cache_mask:
            self.mask_cache = {} if root_mask is not None else None

    def _get_pil_image(self, path):
        if self.cache_mode:
            if path not in self.cache:
                with open(os.path.join(self.root, path), "rb") as f:
                    self.cache[path] = f.read()

            return Image.open(BytesIO(self.cache[path])).convert("RGB")

        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _get_cv2_image(self, path):
        if self.cache_mode:
            if path not in self.cache:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.cache[path] = img

            return Image.fromarray(self.cache[path])

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return Image.fromarray(img)

    def get_image(self, path):
        if self.reader_type == "pil":
            return self._get_pil_image(path)
        elif self.reader_type == "cv2":
            return self._get_cv2_image(path)
        else:
            raise ValueError("Only pil|cv2 supported!")

    def get_mask(self, path):
        if self.cache_mode or self.cache_mask:
            if path not in self.mask_cache:
                loads = np.load(os.path.join(self.root_mask, path))
                self.mask_cache[path] = loads

            return self.mask_cache[path]

        return np.load(os.path.join(self.root_mask, path))

    def __getitem__(self, idx):
        path = self.samples[idx][0]

        if self.mask_type == "fh":
            mask_path = path.replace(self.root, self.root_mask).replace("JPEG", "npy")
            masks = self.get_mask(mask_path)
            masks = torch.from_numpy(masks)
        else:
            masks = None

        sample = self.get_image(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, masks
