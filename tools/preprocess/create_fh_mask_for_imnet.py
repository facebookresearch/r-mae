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
from torchvision.datasets import ImageFolder
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


# def _is_png(filename):
#     """Determine if a file contains a PNG format image.
#     Args:
#         filename: string, path of the image file.
#     Returns:
#         boolean indicating if the image is a PNG.
#     """
#     # File list from:
#     # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
#     return "n02105855_2933.JPEG" in filename


# def _is_cmyk(filename):
#     """Determine if file contains a CMYK JPEG format image.
#     Args:
#         filename: string, path of the image file.
#     Returns:
#         boolean indicating if the image is a JPEG encoded with CMYK color space.
#     """
#     # File list from:
#     # https://github.com/cytsai/ilsvrc-cmyk-image-list
#     cmyk_excluded = [
#         "n01739381_1309.JPEG",
#         "n02077923_14822.JPEG",
#         "n02447366_23489.JPEG",
#         "n02492035_15739.JPEG",
#         "n02747177_10752.JPEG",
#         "n03018349_4028.JPEG",
#         "n03062245_4620.JPEG",
#         "n03347037_9675.JPEG",
#         "n03467068_12171.JPEG",
#         "n03529860_11437.JPEG",
#         "n03544143_17228.JPEG",
#         "n03633091_5218.JPEG",
#         "n03710637_5125.JPEG",
#         "n03961711_5286.JPEG",
#         "n04033995_2932.JPEG",
#         "n04258138_17003.JPEG",
#         "n04264628_27969.JPEG",
#         "n04336792_7448.JPEG",
#         "n04371774_5854.JPEG",
#         "n04596742_4225.JPEG",
#         "n07583066_647.JPEG",
#         "n13037406_4650.JPEG",
#     ]
#     return filename.split("/")[-1] in cmyk_excluded


def _process_image(
    filename,
    fh_scales=[1000],
    fh_min_sizes=[1000],
    dataset_folder=None,
    output_folder=None,
):
    fh_image_path = filename.replace(dataset_folder, output_folder).replace(
        "JPEG", "npy"
    )
    fh_image_folder = os.path.dirname(fh_image_path)
    os.makedirs(fh_image_folder, exist_ok=True)

    image_data = Image.open(filename).convert("RGB")
    image = np.array(image_data)
    fh_segmentations = compute_fh_segmentation(image, fh_scales, fh_min_sizes)

    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    np.save(fh_image_path, fh_segmentations)


def _get_imnet_structure(dataset_folder):
    imnet = ImageFolder(dataset_folder)

    print(f"Pre-processing {len(imnet)} images from ImageNet")
    image_files = []
    for i in range(len(imnet)):
        path = imnet.imgs[i][0]
        image_files.append(path)

    return image_files


def main(args):
    if not os.path.isabs(args.image_folder):
        image_folder = os.path.join(args.root_path, args.image_folder)
    else:
        image_folder = args.image_folder

    if not os.path.exists(image_folder):
        raise RuntimeError("image_folder does not exist")

    if not os.path.isabs(args.output_folder):
        output_folder = os.path.join(args.output_path, args.output_folder)
    else:
        output_folder = args.output_folder

    assert (
        image_folder != output_folder
    ), "image_folder should be different from output_folder"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = _get_imnet_structure(image_folder)
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
                dataset_folder=image_folder,
                output_folder=output_folder,
            ),
            image_files,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", type=str, default="/datasets01/imagenet_full_size/061417"
    )
    parser.add_argument("--image_folder", type=str, default="train")
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--output_folder", type=str, default="fh_train")
    parser.add_argument("--fh_scales", type=str, default="1000")
    parser.add_argument("--fh_min_sizes", type=str, default="1000")
    parser.add_argument("--ntasks", type=int, default=64)

    args = parser.parse_args()
    main(args)
