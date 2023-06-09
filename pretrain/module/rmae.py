# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import warnings
from functools import partial

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mae import MaskedAutoencoderViT
from pretrain.utils.functional import get_2d_sincos_pos_embed
from pretrain.module.layers import (
    DecoderBlockWithExpansion,
    DecoderBlock,
    PatchEmbed,
    Block,
)


class RegionMaskedAutoencoderViT(MaskedAutoencoderViT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_region=0,
        use_mae_loss=True,
        mae_loss_weight=1.0,
        bg_loss_weight=1.0,
        region_loss_weight=1.0,
        region_mask_ratio=0.75,
        region_enc_dim=768,
        region_sample_type="random",
    ):
        if use_mae_loss is False and num_region == 0:
            raise ValueError(
                "There should be at least one loss in training. Found use_mae_loss=False and num_region=0!"
            )

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            use_mae_loss=use_mae_loss,
        )

        self.num_region = num_region
        self.mae_loss_weight = mae_loss_weight
        self.bg_loss_weight = bg_loss_weight
        self.region_loss_weight = region_loss_weight
        self.region_mask_ratio = region_mask_ratio
        self.region_enc_dim = region_enc_dim

        assert region_sample_type in (
            "random",
            "random_fg",
        ), "Only random|random_fg are allowed for region_sample_type"
        self.region_sample_type = region_sample_type

    def random_masking(self, x, mask_ratio, region, shuffle_ids):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        region: [N, num_region, L, region_enc_dim]
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        if self.num_region > 0:
            len_region_keep = int(L * (1 - self.region_mask_ratio))

            if self.region_sample_type == "random":
                noise = torch.rand(N, L, device=x.device)
                # ascend: small is keep, large is remove
                region_shuffle = torch.argsort(noise, dim=1)
            elif self.region_sample_type == "random_fg":
                region_shuffle = shuffle_ids

            region_restore = torch.argsort(region_shuffle, dim=1)
            region_keep = region_shuffle[:, :len_region_keep]
            region_keep = region_keep.unsqueeze(-1).expand(-1, -1, self.region_enc_dim)

            region_mask = torch.ones([N, L], device=x.device)
            region_mask[:, :len_region_keep] = 0
            region_mask = torch.gather(region_mask, dim=1, index=region_restore)

            if region is not None:
                if region_keep.dim() < region.dim():
                    region_keep = region_keep.unsqueeze(1).expand(
                        -1, self.num_region, -1, -1
                    )
                region_masked = torch.gather(region, dim=-2, index=region_keep)
            else:
                region_masked = None
        else:
            region_mask = None
            region_masked = None
            region_restore = None

        if self.num_region > 0:
            ids_shuffle = region_shuffle
            ids_restore = region_restore
        else:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample
            ids_shuffle = torch.argsort(
                noise, dim=1
            )  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, region_masked, region_mask, region_restore

    def forward_encoder(self, x, mask_ratio, shuffle_ids=None, region=None):
        # embed patches
        x = self.patch_embed(x)[0]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        (
            x,
            mask,
            ids_restore,
            region,
            region_mask,
            region_restore,
        ) = self.random_masking(x, mask_ratio, region, shuffle_ids)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)[0]
        x = self.norm(x)

        return x, mask, ids_restore, region, region_mask, region_restore

    def forward_region_decoder(self):
        raise NotImplementedError

    def _forward_region_loss(self, pred, pred_target):
        region_loss = F.binary_cross_entropy_with_logits(
            pred, pred_target, reduction="none"
        )

        if self.bg_loss_weight != 1.0:
            weight_loss = pred_target.detach().clone()
            weight_loss[pred_target == 0] = self.bg_loss_weight
            region_loss = region_loss * weight_loss
        region_loss = region_loss.mean(dim=-1)

        return region_loss

    def forward_loss(self, imgs, pred, mask, pred_region, region_mask, target_region):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if self.use_mae_loss:
            target = self.patchify(imgs)

            if self.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.0e-6) ** 0.5

            mae_loss = self._forward_mae_loss(pred, target)
        else:
            mae_loss = torch.zeros_like(mask)

        if self.num_region > 0:
            region_loss = self._forward_region_loss(pred_region, target_region)
        else:
            region_loss = torch.zeros_like(mae_loss)
            region_mask = torch.ones_like(mask)

        mae_loss = (mae_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        region_loss = (
            region_loss * region_mask
        ).sum() / region_mask.sum()  # mean loss on removed patches
        loss = mae_loss * self.mae_loss_weight + region_loss * self.region_loss_weight

        return loss, mae_loss, region_loss

    def forward(self):
        raise NotImplementedError


class RegionQueryRMAE(RegionMaskedAutoencoderViT):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_region=0,
        use_mae_loss=True,
        mae_loss_weight=1.0,
        bg_loss_weight=1.0,
        region_loss_weight=1.0,
        region_mask_ratio=0.75,
        region_enc_dim=768,
        region_enc_depth=1,
        region_enc_num_heads=8,
        region_dec_dim=128,
        region_dec_depth=1,
        region_dec_num_heads=8,
        region_sample_type="random",
        region_cross_layer=8,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth if use_mae_loss else region_cross_layer,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            num_region=num_region,
            use_mae_loss=use_mae_loss,
            mae_loss_weight=mae_loss_weight,
            bg_loss_weight=bg_loss_weight,
            region_loss_weight=region_loss_weight,
            region_mask_ratio=region_mask_ratio,
            region_enc_dim=region_enc_dim,
            region_sample_type=region_sample_type,
        )

        self.use_mae_loss = use_mae_loss
        if use_mae_loss is False:
            self.decoder_norm = None
            self.decoder_pred = None

        # --------------------------------------------------------------------------
        # MAE region specifics
        if num_region > 0:
            num_patches = self.patch_embed.num_patches
            self.region_cross_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, region_dec_dim),
                requires_grad=False,
            )  # fixed sin-cos embedding

            self.region_cross_embed = nn.Linear(embed_dim, region_dec_dim, bias=True)
            self.region_dec_norm = norm_layer(region_dec_dim)
            # self.region_dec_norm = nn.Identity()
            self.region_cross_mask_token = nn.Parameter(
                torch.zeros(1, 1, region_dec_dim)
            )
            self.region_cross_blocks = nn.ModuleList()
            for _ in range(region_cross_layer):
                self.region_cross_blocks.append(
                    Block(
                        region_dec_dim,
                        region_dec_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                )
            cross_embed_dim = region_dec_dim

            self.region_patch_embed = PatchEmbed(
                img_size, patch_size, 1, region_enc_dim
            )
            self.region_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, region_enc_dim), requires_grad=False
            )  # fixed sin-cos embedding
            self.region_enc_blocks = nn.ModuleList(
                [
                    Block(
                        region_enc_dim,
                        region_enc_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for i in range(region_enc_depth)
                ]
            )

            self.region_proj = nn.Sequential(
                norm_layer(region_enc_dim),
                nn.Linear(region_enc_dim, region_dec_dim),
            )

            self.region_dec_blocks = nn.ModuleList()

            for _ in range(region_dec_depth - 1):
                self.region_dec_blocks.append(
                    DecoderBlock(
                        region_dec_dim,
                        cross_embed_dim,
                        region_dec_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                )
            self.region_dec_blocks.append(
                DecoderBlockWithExpansion(
                    region_dec_dim,
                    cross_embed_dim,
                    region_dec_num_heads,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
            )
            self.region_pred = nn.Sequential(
                nn.GELU(),
                nn.Linear(region_dec_dim, patch_size**2, bias=True),
            )
        self.region_cross_layer = region_cross_layer

        self.initialize_rmae_weights()

    def initialize_rmae_weights(self):
        super().initialize_weights()

        if self.num_region > 0:
            region_pos_embed = get_2d_sincos_pos_embed(
                self.region_pos_embed.shape[-1],
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )
            self.region_pos_embed.data.copy_(
                torch.from_numpy(region_pos_embed).float().unsqueeze(0)
            )

            # initialize region_patch_embed like nn.Linear (instead of nn.Conv2d)
            w = self.region_patch_embed.proj.weight.data
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

            torch.nn.init.normal_(self.region_cross_mask_token, std=0.02)

            region_cross_pos_embed = get_2d_sincos_pos_embed(
                self.region_cross_pos_embed.shape[-1],
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )
            self.region_cross_pos_embed.data.copy_(
                torch.from_numpy(region_cross_pos_embed).float().unsqueeze(0)
            )

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward_encoder(self, x, mask_ratio, shuffle_ids=None, region=None):
        if self.num_region > 0:
            region = region + self.region_pos_embed[None, :, 1:, :]

        return super().forward_encoder(
            x,
            mask_ratio,
            shuffle_ids=shuffle_ids,
            region=region,
        )

    def forward_decoder(self, x, ids_restore):
        x_cross = None
        if self.num_region > 0:
            x_cross = self.region_cross_embed(x)
            cross_mask_tokens = self.region_cross_mask_token.expand(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1
            )
            x_cross_ = torch.cat([x_cross[:, 1:, :], cross_mask_tokens], dim=1)
            x_cross_ = torch.gather(
                x_cross_,
                dim=1,
                index=ids_restore.unsqueeze(-1).expand(-1, -1, x_cross.shape[2]),
            )
            x_cross = torch.cat([x_cross[:, :1, :], x_cross_], dim=1)

            x_cross = x_cross + self.region_cross_pos_embed

            for blk in self.region_cross_blocks:
                x_cross = blk(x_cross)[0]
            x_cross = self.region_dec_norm(x_cross)

        if self.use_mae_loss:
            # embed tokens
            x = self.decoder_embed(x)

            # append mask tokens to sequence
            mask_tokens = self.mask_token.expand(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1
            )
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
            )  # unshuffle

            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
            # add pos embed
            x = x + self.decoder_pos_embed

            # apply Transformer blocks
            for i, blk in enumerate(self.decoder_blocks):
                x = blk(x)[0]

        if self.use_mae_loss:
            x = self.decoder_norm(x)

            # predictor projection
            x = self.decoder_pred(x)

            # remove cls token
            x = x[:, 1:, :]
        else:
            x = None

        return x, x_cross

    def forward_region_encoder(self, region):
        l = region.shape[1]
        for blk in self.region_enc_blocks:
            region = blk(region)[0]
        region = region.view(-1, self.num_region, l, self.region_enc_dim)

        region = region.mean(dim=2)

        return region

    def forward_region_decoder(self, region, memory):
        region = self.region_proj(region)
        for blk in self.region_dec_blocks:
            region = blk(region, memory)[0]

        region = self.region_pred(region)
        region = region[:, :, 1:, :]

        return region

    def _forward_region_loss(self, pred, pred_target):
        region_loss = super()._forward_region_loss(pred, pred_target)
        region_loss = region_loss.mean(dim=1)

        return region_loss

    def forward(self, imgs, mask_ratio=0.75, region=None, shuffle_ids=None):
        if region is not None:
            b, c, h, w = region.shape
            region = region.view(b * c, 1, h, w)
            target_region = self.patchify(region).view(b, c, -1, self.patch_size**2)
            region = self.region_patch_embed(region - 0.5)[0]
            region = region.view(b, c, -1, self.region_enc_dim)
        else:
            region = None
            target_region = None

        (
            latent,
            mask,
            ids_restore,
            region_masked,
            region_mask,
            _,
        ) = self.forward_encoder(
            imgs,
            mask_ratio,
            shuffle_ids=shuffle_ids,
            region=region,
        )

        if region is not None:
            region_latent = self.forward_region_encoder(region_masked.flatten(0, 1))
        else:
            region_latent = None

        pred, memory = self.forward_decoder(latent, ids_restore)

        if self.num_region > 0:
            pred_region = self.forward_region_decoder(region_latent, memory)
        else:
            pred_region = None

        loss, mae_loss, region_loss = self.forward_loss(
            imgs, pred, mask, pred_region, region_mask, target_region
        )

        return loss, (pred, pred_region), (mae_loss, region_loss), (mask, region_mask)


def build_rmae(config):
    arch = config["arch"]
    params = copy.deepcopy(config["params"])

    with omegaconf.open_dict(params):
        if "rmae_base_patch16" in arch:
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
        elif "rmae_large_patch16" in arch:
            patch_size = 16
            embed_dim = 1024
            depth = 24
            num_heads = 16
        elif "rmae_huge_patch14" in arch:
            patch_size = 14
            embed_dim = 1280
            depth = 32
            num_heads = 16
        else:
            raise ValueError(
                "Only support rmae_base_patch16|rmae_large_patch16|rmae_huge_patch14"
            )

    net = RegionQueryRMAE(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **params,
    )

    return net
