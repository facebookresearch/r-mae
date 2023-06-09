# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

from setuptools import find_packages, setup

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
assert TORCH_VERSION >= (1, 8), "Requires PyTorch >= 1.8"


def get_version():
    init_py_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "pretrain", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    return version


setup(
    name="pretrain",
    version=get_version(),
    author="Duy-Kien Nguyen",
    description="Implementation for R-MAE.",
    packages=find_packages(exclude=("tests", "exps", "scripts")),
    python_requires=">=3.8",
    install_requires=[
        "Pillow>=7.1",
        "omegaconf>=2.1",
        "pycocotools",
        "numpy",
        "matplotlib",
        "scikit-image",
        "opencv-python",
        "tensorboard",
    ],
)
