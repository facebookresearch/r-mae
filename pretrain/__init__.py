# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pretrain.utils
import pretrain.model
import pretrain.optim
import pretrain.trainer
import pretrain.dataset

from .utils.env import setup_environment

setup_environment()

__version__ = "0.1"
