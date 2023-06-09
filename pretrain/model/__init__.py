# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os

from pretrain.model.base_model import BaseModel


ARCH_REGISTRY = {}


__all__ = ["BaseModel"]


def build_model(config, num_classes):
    model_name = config.model
    model_config = config.model_config[model_name]

    if model_name not in ARCH_REGISTRY:
        raise ValueError("Model architecture ({}) is not found.".format(model_name))
    if num_classes is not None:
        model = ARCH_REGISTRY[model_name](
            model_config, num_classes, global_config=config
        )
    else:
        model = ARCH_REGISTRY[model_name](model_config, global_config=config)

    if hasattr(model, "build"):
        model.build()

    return model


def register_model(name):
    def register_model_cls(cls):
        if name in ARCH_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        elif not issubclass(cls, BaseModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseModel".format(name, cls.__name__),
            )

        ARCH_REGISTRY[name] = cls
        return cls

    return register_model_cls


def get_arch_list():
    return tuple(ARCH_REGISTRY.keys())


models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    path = os.path.join(models_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("pretrain.model." + model_name)
