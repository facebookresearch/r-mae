# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os


TRAINER_REGISTRY = {}


def build_trainer(configuration, *args, **kwargs):
    configuration.freeze()

    config = configuration.get_config()
    trainer = config.training.trainer
    trainer = TRAINER_REGISTRY[trainer](configuration)

    return trainer


def register_trainer(name):
    def register_trainer_cls(cls):
        if name in TRAINER_REGISTRY:
            raise ValueError("Cannot register duplicate trainer ({})".format(name))

        TRAINER_REGISTRY[name] = cls
        return cls

    return register_trainer_cls


trainers_dir = os.path.dirname(__file__)
for file in os.listdir(trainers_dir):
    path = os.path.join(trainers_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        trainer_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("pretrain.trainer." + trainer_name)
