# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

from pretrain.model import BaseModel, register_model
from pretrain.module import build_mae, build_rmae
from pretrain.utils.modeling import get_mae_parameters


@register_model("mae")
class MaskedAutoencoder(BaseModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, global_config=kwargs["global_config"])
        self.mask_ratio = config["mask_ratio"]

    def get_optimizer_parameters(self):
        wd_norm = self._global_config.optimizer.params.wd_norm
        wd_bias = self._global_config.optimizer.params.wd_bias

        mae_groups = get_mae_parameters(self.mae_vit, wd_norm=wd_norm, wd_bias=wd_bias)

        return mae_groups

    def _build(self):
        mae_vit_config = copy.deepcopy(self.config.mae_vit)
        self.mae_vit = build_mae(mae_vit_config)

    def state_dict(self, *args, **kwargs):
        return self.mae_vit.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.mae_vit.load_state_dict(state_dict, *args, **kwargs)

    def forward(self, sample, target=None):
        imgs = sample["image"]

        loss, pred, metric, mask = self.mae_vit(imgs, self.mask_ratio)

        if not self.training:
            return loss, pred, mask

        return {"losses": {"pretrain_loss": loss}, "metrics": {"mae_loss": metric}}


@register_model("rmae")
class RegionMaskedAutoencoder(MaskedAutoencoder):
    def __init__(self, config, **kwargs):
        super().__init__(config, global_config=kwargs["global_config"])
        self.num_region = config.mae_vit.params.num_region
        self.region_sample_type = config.mae_vit.params.region_sample_type

    def _build(self):
        mae_vit_config = copy.deepcopy(self.config.mae_vit)
        self.mae_vit = build_rmae(mae_vit_config)

    def forward(self, sample, target=None):
        imgs = sample["image"]

        if self.num_region > 0:
            region = sample["region"]
        else:
            region = None

        if self.region_sample_type == "random_fg":
            assert self.num_region > 0, "num_region should be greater than 0"
            shuffle_ids = sample["shuffle_ids"]
        else:
            shuffle_ids = None

        loss, pred, metric, mask = self.mae_vit(
            imgs, self.mask_ratio, region=region, shuffle_ids=shuffle_ids
        )

        if not self.training:
            return loss, pred[0], pred[1], region, mask

        mae_loss, proposal_loss = metric

        return {
            "losses": {"pretrain_loss": loss},
            "metrics": {"mae_loss": mae_loss, "region_loss": proposal_loss},
        }
