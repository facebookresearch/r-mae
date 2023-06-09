# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Set, Dict, Any

import torch


def get_layer_id(layer_name, num_layers):
    if "net.pos_embed" in layer_name:
        return 0
    elif "net.patch_embed" in layer_name:
        return 0
    elif "net.blocks." in layer_name:
        layer_id = int(layer_name[layer_name.find("net.blocks.") :].split(".")[2])
        return layer_id + 1

    return num_layers - 1


norm_module_types = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    # NaiveSyncBatchNorm inherits from BatchNorm2d
    torch.nn.GroupNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.LayerNorm,
    torch.nn.LocalResponseNorm,
)


def get_parameters(
    model: torch.nn.Module,
    lr_multi: Optional[float] = 1.0,
    lr_module: Optional[List[str]] = [],
    wd_norm: Optional[float] = None,
    module_except: Optional[List[str]] = [],
):
    param_group_wd_norm = {"params": []}
    param_group_lr_multi = {"params": []}
    param_group_others = {"params": []}

    for module_name, module in model.named_modules():
        if any(nd in module_name for nd in module_except):
            continue

        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, norm_module_types):
                param_group_wd_norm["params"].append(param)
            elif any(nd in param_name for nd in lr_module):
                param_group_lr_multi["params"].append(param)
            else:
                param_group_others["params"].append(param)

    if lr_multi is not None and lr_multi != 1.0:
        param_group_lr_multi["lr_multi"] = lr_multi

    if wd_norm is not None:
        param_group_wd_norm["weight_decay"] = wd_norm

    optimizer_grouped_parameters = [
        param_group_wd_norm,
        param_group_lr_multi,
        param_group_others,
    ]

    return optimizer_grouped_parameters


def get_vit_parameters(
    model: torch.nn.Module,
    wd_except: Optional[List[str]] = None,
    wd_bias: Optional[float] = None,
    wd_norm: Optional[float] = None,
    lr_decay_rate: Optional[float] = None,
    num_layers: Optional[int] = None,
):
    memo: Set[torch.nn.parameter.Parameter] = set()

    if lr_decay_rate is not None:
        assert num_layers is not None
        num_layers += 2

    if lr_decay_rate is not None:
        param_group_decay = [{"params": []} for _ in range(num_layers + 1)]
        param_group_no_decay = [
            {"params": [], "weight_decay": 0.0} for _ in range(num_layers + 1)
        ]
    else:
        param_group_decay = [{"params": []}]
        param_group_no_decay = [{"params": [], "weight_decay": 0.0}]

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param in memo:
                continue
            memo.add(param)

            no_decay = False
            if isinstance(module, norm_module_types) and wd_norm == 0.0:
                no_decay = True

            if "bias" in param_name and wd_bias == 0.0:
                no_decay = True

            if wd_except is not None and any(nd in param_name for nd in wd_except):
                no_decay = True

            if lr_decay_rate is not None:
                layer_id = get_layer_id(f"{module_name}.{param_name}", num_layers)
                if no_decay:
                    param_group_no_decay[layer_id]["params"].append(param)
                    param_group_no_decay[layer_id]["lr_multi"] = lr_decay_rate ** (
                        num_layers - 1 - layer_id
                    )
                else:
                    param_group_decay[layer_id]["params"].append(param)
                    param_group_decay[layer_id]["lr_multi"] = lr_decay_rate ** (
                        num_layers - 1 - layer_id
                    )
            else:
                if no_decay:
                    param_group_no_decay[0]["params"].append(param)
                else:
                    param_group_decay[0]["params"].append(param)
    optimizer_grouped_parameters = param_group_decay + param_group_no_decay

    return optimizer_grouped_parameters


def get_mae_parameters(
    model: torch.nn.Module,
    wd_except: Optional[List[str]] = None,
    wd_bias: Optional[float] = None,
    wd_norm: Optional[float] = None,
):
    memo: Set[torch.nn.parameter.Parameter] = set()

    param_group_decay = {"params": []}
    param_group_no_decay = {"params": [], "weight_decay": 0.0}

    for module in model.modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if param in memo:
                continue
            memo.add(param)

            no_decay = False
            if isinstance(module, norm_module_types) and wd_norm == 0.0:
                no_decay = True

            if "bias" in param_name and wd_bias == 0.0:
                no_decay = True

            if wd_except is not None and any(nd in param_name for nd in wd_except):
                no_decay = True

            if no_decay:
                param_group_no_decay["params"].append(param)
            else:
                param_group_decay["params"].append(param)
    optimizer_grouped_parameters = [param_group_decay, param_group_no_decay]

    return optimizer_grouped_parameters
