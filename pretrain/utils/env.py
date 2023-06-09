# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pretrain.dataset import get_task_list
from pretrain.model import get_arch_list
from pretrain.utils.distributed import get_rank


def set_seed(seed):
    if seed:
        if seed == -1:
            # From detectron2
            seed = (
                os.getpid()
                + int(datetime.datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
            )
        else:
            seed = seed + get_rank()
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        cudnn.benchmark = True

    return seed


def _configure_libraries():
    # Disable opencl in opencv since its interaction with cuda often has negative effects
    # This envvar is supported after OpenCV 3.4.0
    os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
    try:
        import cv2

        if int(cv2.__version__.split(".")[0]) >= 3:
            cv2.ocl.setUseOpenCL(False)
    except ModuleNotFoundError:
        # Other types of ImportError, if happened, should not be ignored.
        # Because a failed opencv import could mess up address space
        # https://github.com/skvark/opencv-python/issues/381
        pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split(".")[:digit]))

    import yaml

    assert get_version(yaml) >= (5, 1), "Requires pyyaml>=5.1"
    assert get_version(torch) >= (1, 4), "Requires torch>=1.4"


_ENV_SETUP_DONE = False


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $DETECTRON2_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument_group("Core Arguments")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=get_arch_list(),
        help="The architecture for training and testing.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=get_task_list(),
        help="The working task.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The corresponding dataset.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs="*",
        help="Modify config options from command line",
    )

    return parser
