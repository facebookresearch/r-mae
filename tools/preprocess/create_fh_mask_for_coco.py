# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import glob
from functools import partial
from multiprocessing import Pool

import numpy as np
import skimage.segmentation
from PIL import Image


def compute_fh_segmentation(image_np, scales, min_sizes):
    """Compute FSZ segmentation on image and record stats."""
    fh_segmentations = []
    for scale, min_size in zip(scales, min_sizes):
        segmented_image = skimage.segmentation.felzenszwalb(
            image_np, scale=scale, min_size=min_size
        )
        segmented_image = segmented_image.astype(np.dtype("<u1"))
        fh_segmentations.append(segmented_image)
    fh_segmentations = np.stack(fh_segmentations)

    return fh_segmentations


def _process_image(filename, fh_scales=[1000], fh_min_sizes=[1000], output_folder=None):
    image_name = filename.split("/")[-1].replace("jpg", "npy")
    fh_image_path = os.path.join(output_folder, image_name)

    image_data = Image.open(filename).convert("RGB")
    image = np.array(image_data)
    fh_segmentations = compute_fh_segmentation(image, fh_scales, fh_min_sizes)

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    np.save(fh_image_path, fh_segmentations)


def main(args):
    if not os.path.isabs(args.image_folder):
        image_folder = os.path.join(args.root_path, args.image_folder)
    else:
        image_folder = args.image_folder

    if not os.path.exists(image_folder):
        raise RuntimeError("image_folder does not exist")

    if not os.path.isabs(args.output_folder):
        output_folder = os.path.join(args.root_path, args.output_folder)
    else:
        output_folder = args.output_folder

    assert (
        image_folder != output_folder
    ), "image_folder should be different from output_folder"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(image_folder + "/*.jpg")
    fh_scales = [int(n) for n in args.fh_scales.split(",")]
    fh_min_sizes = [int(n) for n in args.fh_min_sizes.split(",")]
    print("fh_scales:", fh_scales)
    print("fh_min_scales:", fh_min_sizes)

    with Pool(args.ntasks) as p:
        p.map(
            partial(
                _process_image,
                fh_scales=fh_scales,
                fh_min_sizes=fh_min_sizes,
                output_folder=output_folder,
            ),
            image_files,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="./")
    parser.add_argument("--image_folder", type=str, default="train2017")
    parser.add_argument("--output_folder", type=str, default="fh_train2017")
    parser.add_argument("--fh_scales", type=str, default="1000")
    parser.add_argument("--fh_min_sizes", type=str, default="1000")
    parser.add_argument("--ntasks", type=int, default=32)

    args = parser.parse_args()
    main(args)
