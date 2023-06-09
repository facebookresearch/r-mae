# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import collections.abc
from lib2to3.pgen2.token import OP
import os
import copy

import torch
import torch.optim as optim
import omegaconf

from pretrain.optim.oss import OSS
from pretrain.optim.lars import LARS
from pretrain.utils.general import get_optimizer_parameters


OPTIM_REGISTRY = {"sgd": optim.SGD, "adamw": optim.AdamW, "lars": LARS}


def build_optimizer(config, model):
    optim_type = config.optimizer["type"]
    optim_config = copy.deepcopy(config.optimizer["params"])

    with omegaconf.open_dict(optim_config):
        use_oss = optim_config.pop("use_oss", False)
        redundants = ["lr_decay_rate", "wd_norm", "wd_bias"]
        for redundant in redundants:
            if redundant in optim_config:
                optim_config.pop(redundant)

    if optim_type not in OPTIM_REGISTRY:
        raise ValueError("Optimizer ({}) is not found.".format(optim_type))

    model_params = get_optimizer_parameters(model)

    if isinstance(model_params[0], collections.abc.Sequence):
        param_groups = []
        backbone_group, transformer_group = model_params

        with omegaconf.open_dict(optim_config):
            lr_backbone = optim_config.pop("lr_backbone", optim_config["lr"])

            for bgroup in backbone_group:
                if "lr_multi" in bgroup:
                    bgroup["lr"] = lr_backbone * bgroup.pop("lr_multi")
                else:
                    bgroup["lr"] = lr_backbone
                param_groups.append(bgroup)

            for tgroup in transformer_group:
                if "lr_multi" in tgroup:
                    tgroup["lr"] = optim_config["lr"] * tgroup.pop("lr_multi")
                param_groups.append(tgroup)
    elif isinstance(model_params[0], collections.abc.Mapping):
        param_groups = model_params
    else:
        param_groups = [{"lr": optim_config["lr"], "params": model_params}]

    if use_oss:
        optimizer = OSS(
            params=param_groups, optim=OPTIM_REGISTRY[optim_type], **optim_config
        )
    else:
        optimizer = OPTIM_REGISTRY[optim_type](param_groups, **optim_config)

    return optimizer


def register_optim(name):
    def register_optim_cls(cls):
        if name in OPTIM_REGISTRY:
            raise ValueError("Cannot register duplicate optimizer ({})".format(name))
        elif not issubclass(cls, torch.optim.Optimizer):
            raise ValueError(
                "Optimizer ({}: {}) must extend torch.optim.Optimizer".format(
                    name, cls.__name__
                )
            )

        OPTIM_REGISTRY[name] = cls
        return cls

    return register_optim_cls


optims_dir = os.path.dirname(__file__)
for file in os.listdir(optims_dir):
    path = os.path.join(optims_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        optim_name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("pretrain.optim." + optim_name)
