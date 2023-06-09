# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import collections

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

from pretrain.dataset.helper import Prefetcher
from pretrain.trainer.engine import register_engine
from pretrain.utils.distributed import is_master, synchronize
from pretrain.utils.functional import patchify, unpatchify
from pretrain.trainer.engine.base_engine import BaseEngine


@register_engine("pretrain")
class PretrainEngine(BaseEngine):
    def __init__(self, trainer):
        super().__init__(trainer)

    def _compute_loss(self, model_output, target):
        if self.model.training:
            iter_per_update = self.trainer.iter_per_update
            # Make sure theat the output from the model is a Mapping
            assert isinstance(
                model_output, collections.abc.Mapping
            ), "A dict must be returned from the forward of the model."

            assert "losses" in model_output, "losses should be returned in model_output"
            assert isinstance(
                model_output["losses"], collections.abc.Mapping
            ), "'losses' must be a dict."

            loss_dict = {
                k: v / iter_per_update for k, v in model_output["losses"].items()
            }

            losses_stat = {}

            total_loss = sum(loss_dict[k] for k in loss_dict.keys())
            losses_stat.update({k: v for k, v in loss_dict.items()})

            losses_stat["total_loss"] = total_loss
            model_output["losses"] = total_loss
            model_output["losses_stat"] = losses_stat

            assert (
                "metrics" in model_output
            ), "metrics should be returned in model_output"
            assert isinstance(
                model_output["metrics"], collections.abc.Mapping
            ), "'metrics' must be a dict."

            for kk, vv in model_output["metrics"].items():
                model_output["metrics"][kk] = vv.detach()

        return model_output

    @torch.no_grad()
    def evaluate(self, split):
        self.trainer.writer.write(f"Evaluation time. Running on full {split} set...")
        self.trainer.timers[split].reset()

        self.model.eval()

        invTrans = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                ),
                transforms.Normalize(
                    mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]
                ),
                # transforms.ToPILImage(),
            ]
        )

        prefetcher = Prefetcher(
            self.dataloaders[split], self.datasets[split], prefetch=False
        )

        for i in range(15):
            batch = prefetcher.get_next_sample()
            output = self._forward(batch)[0]

            sample = batch[0]
            _, pred, pred_region, mask_label, (mask, region_mask) = output

            b, c, h, w = sample["image"].shape
            num_region = (
                0
                if not getattr(self.datasets[split], "rmae_sampling", False)
                else self.datasets[split].num_region
            )

            if num_region > 0:
                patch_size = self.datasets[split].patch_size
                num_patches = h // patch_size

                if region_mask.dim() < pred_region.dim() - 1:
                    pred_region = torch.where(
                        ~(region_mask.unsqueeze(-1).unsqueeze(1).bool()),
                        torch.tensor(
                            [0.5], dtype=torch.float, device=pred_region.device
                        ),
                        (pred_region >= 0).float(),
                    )
                else:
                    pred_region = torch.where(
                        ~(region_mask.unsqueeze(-1).bool()),
                        torch.tensor(
                            [0.5], dtype=torch.float, device=pred_region.device
                        ),
                        (pred_region >= 0).float(),
                    )
                pred_region = pred_region.view(
                    b, -1, num_patches, num_patches, patch_size, patch_size
                )

                pred_region = pred_region.permute(0, 1, 2, 4, 3, 5).reshape(b, -1, h, w)

            patches = patchify(sample["image"], patch_size)

            masked_image = patches.masked_fill(mask.unsqueeze(-1).bool(), 0.0)
            masked_image = unpatchify(masked_image, patch_size=patch_size)

            if pred is not None:
                mean = patches.mean(dim=-1, keepdim=True)
                var = patches.var(dim=-1, keepdim=True)
                pred = pred * ((var + 1.0e-6) ** 0.5) + mean
                pred_image = unpatchify(pred, patch_size=patch_size)
                pred_masked_image = unpatchify(
                    torch.where(mask.unsqueeze(-1).bool(), pred, patches),
                    patch_size=patch_size,
                )

            image_i = invTrans(sample["image"][0].cpu()).clamp(min=0, max=1)
            masked_image_i = invTrans(masked_image[0].cpu()).clamp(min=0, max=1)

            image_vis = make_grid([image_i, masked_image_i], nrow=2, pad_value=1.0)

            self.trainer.tb_writer.add_image(
                f"{i}_image", image_vis, self.current_epoch
            )

            if pred is not None:
                pred_image_i = invTrans(pred_image[0].cpu()).clamp(min=0, max=1)
                pred_masked_image_i = invTrans(pred_masked_image[0].cpu()).clamp(
                    min=0, max=1
                )

                pred_image_vis = make_grid(
                    [pred_image_i, pred_masked_image_i], nrow=2, pad_value=1.0
                )

                self.trainer.tb_writer.add_image(
                    f"{i}_pred_image", pred_image_vis, self.current_epoch
                )

            if pred_region is not None:
                for j in range(pred_region.shape[1]):
                    region = make_grid(
                        [
                            mask_label[0, [j, j, j]].cpu().clamp(min=0, max=1),
                            pred_region[0, [j, j, j]].cpu().clamp(min=0, max=1),
                        ],
                        nrow=2,
                        pad_value=1.0,
                    )

                    self.trainer.tb_writer.add_image(
                        f"{i}_pred_mask_{j}",
                        region,
                        self.current_epoch,
                    )

        self.model.train()
        gc.collect()
        if "cuda" in str(self.trainer.device):
            torch.cuda.empty_cache()

        self.trainer.timers["train"].reset()

    def train_epoch(self, trained_batch_idx):
        current_epoch = self.trainer.current_epoch
        current_update = self.trainer.current_update
        max_update = self.trainer.max_update
        iter_per_update = self.trainer.iter_per_update
        eval_interval = self.trainer.eval_interval
        save_interval = self.trainer.save_interval

        prefetcher = Prefetcher(
            self.trainer.dataloaders["train"],
            self.trainer.datasets["train"],
            prefetch=False,
        )

        if self.trainer.samplers["train"] is not None and self.trainer.parallel:
            self.trainer.samplers["train"].set_epoch(current_epoch)

        for idx in range(len(self.trainer.dataloaders["train"])):
            # for idx, batch in enumerate(trainer.dataloaders["train"]):
            batch = prefetcher.get_next_sample()
            if idx < trained_batch_idx:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            # for param in self.params:
            #     param.grad = None

            if iter_per_update > 1:
                assert iter_per_update == len(batch)
                for idx, splitted_batch in enumerate(batch):
                    if (idx + 1) < iter_per_update:
                        with self.model.no_sync():
                            output = self._forward(splitted_batch)[0]
                            if output is None:
                                continue
                            self._sync_losses_and_metrics("train", output)
                            self._backward(output)
                    else:
                        output = self._forward(splitted_batch)[0]
                        if output is None:
                            continue
                        self._sync_losses_and_metrics("train", output)
                        self._backward(output)
            else:
                output = self._forward(batch)[0]
                self._sync_losses_and_metrics("train", output)
                self._backward(output)
            current_update = self._step(current_update)

            if current_update == self.trainer.current_update:
                self.trainer.writer.write("Skipping iteration...", "warning")
                continue

            self.lr_scheduler.step(current_update)

            assert self.trainer.current_update == (current_update - 1)
            self.trainer.current_update = current_update
            self._update_info("train")

            if current_update % save_interval == 0:
                self.trainer.writer.write("Checkpoint time. Saving a checkpoint...")
                self.trainer.checkpoint.save(current_update)

            if current_update % eval_interval == 0 and "val" in self.trainer.run_type:
                self.evaluate("val")

            if current_update > max_update:
                break

        self.lr_scheduler.step_epoch(current_epoch)
