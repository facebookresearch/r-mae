# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from pretrain.utils.functional import drop_path


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if isinstance(drop, tuple):
            drop_probs = drop
        else:
            drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()

        if isinstance(img_size, tuple):
            img_size = img_size
        else:
            img_size = (img_size, img_size)

        if isinstance(patch_size, tuple):
            patch_size = patch_size
        else:
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        bchw = x.shape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, bchw


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_cls_token=True,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.use_cls_token = use_cls_token
        self.qkv_bias = qkv_bias

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        # attn_mask: B x N
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), -65504.0)

        if rel_pos_bias is not None:
            if not self.use_cls_token:
                rel_pos_bias = rel_pos_bias[:, 1:, 1:]
            attn = attn + rel_pos_bias

        attn_wo_softmax = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention = attn

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_wo_softmax


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        win_size=0,
        use_cls_token=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cls_token=use_cls_token,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.win_size = win_size
        self.hw = None

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        ori_x = x
        x = self.norm1(x)
        x, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)

        x = ori_x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn


class CrossAttention(nn.Module):
    def __init__(
        self,
        qdim,
        kvdim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_cls_token=True,
    ):
        super().__init__()
        assert qdim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = qdim // num_heads
        self.scale = head_dim**-0.5
        self.use_cls_token = use_cls_token
        self.qkv_bias = qkv_bias

        self.q = nn.Linear(qdim, qdim, bias=qkv_bias)
        self.kv = nn.Linear(kvdim, qdim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(qdim, qdim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, query, memory, rel_pos_bias=None, attn_mask=None, with_mask=False
    ):
        # attn_mask: B x N
        B, N, C = query.shape
        L = memory.shape[1]

        q = (
            self.q(query)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k, v = (
            self.kv(memory)
            .reshape(B, L, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask.unsqueeze(1), -65504.0)

        if rel_pos_bias is not None:
            if not self.use_cls_token:
                rel_pos_bias = rel_pos_bias[:, 1:, 1:]
            attn = attn + rel_pos_bias

        attn_wo_softmax = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        self.attention = attn

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn_wo_softmax


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        enc_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_cls_token=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cls_token=use_cls_token,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            enc_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cls_token=use_cls_token,
        )

        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, memory, rel_pos_bias=None, attn_mask=None):
        ori_x = x
        x = self.norm1(x)
        x, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        x = ori_x + self.drop_path(x)

        ori_x = x
        x = self.norm2(x)
        x, attn = self.cross_attn(
            x,
            memory,
            rel_pos_bias=rel_pos_bias,
            attn_mask=attn_mask,
        )

        x = ori_x + self.drop_path(x)

        ori_x = x
        x = self.drop_path(self.mlp(self.norm3(x)))
        x = ori_x + self.drop_path(x)

        return x, attn


class DecoderBlockWithExpansion(nn.Module):
    def __init__(
        self,
        dim,
        enc_dim,
        num_heads,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_cls_token=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cls_token=use_cls_token,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.Identity()
        self.proj = nn.Linear(enc_dim, dim)

        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, memory, rel_pos_bias=None, attn_mask=None):
        ori_x = x
        x = self.norm1(x)
        x, attn = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
        x = ori_x + self.drop_path(x)

        ori_x = x
        x = self.norm2(x)
        x_mask = self.proj(memory)
        x_mask = ori_x.unsqueeze(2) + self.drop_path(x_mask).unsqueeze(1)

        x_mask = self.mlp(self.norm3(x_mask))

        return x_mask, attn
