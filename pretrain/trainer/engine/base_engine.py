# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from pretrain.utils.distributed import reduce_dict
from pretrain.utils.general import clip_grad_norm, filter_grads


class BaseEngine:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = self.trainer.model
        self.optimizer = self.trainer.optimizer
        self.lr_scheduler = self.trainer.lr_scheduler
        self.dataloaders = self.trainer.dataloaders
        self.datasets = self.trainer.datasets
        self.params = filter_grads(self.model.parameters())
        self.num_skip = 0

    @torch.no_grad()
    def evaluate(self, split):
        raise NotImplementedError

    @property
    def current_epoch(self):
        current_update = self.trainer.current_update
        batch_size = self.trainer.running_config.batch_size
        if self.datasets["train"] is not None:
            update_per_epoch = len(self.datasets["train"]) // batch_size
        else:
            update_per_epoch = 1

        return (current_update + update_per_epoch - 1) // update_per_epoch

    def train_epoch(self, trained_batch_idx):
        raise NotImplementedError

    def _compute_loss(self, output, target):
        raise NotImplementedError

    def _forward(self, batch, **kwargs):
        self.trainer.profile("Batch prepare time")

        sample, target = batch

        if self.trainer.use_fp16:
            assert self.trainer.use_fp16 in ("float16", "bfloat16")
            dtype = (
                torch.bfloat16 if self.trainer.use_fp16 == "bfloat16" else torch.float16
            )
            with torch.autocast(device_type="cuda", dtype=dtype):
                output = self.model(sample, target)
                output = self._compute_loss(output, target, **kwargs)
        else:
            output = self.model(sample, target)
            output = self._compute_loss(output, target, **kwargs)
        self.trainer.profile("Forward time")

        return output, target

    def _backward(self, output):
        loss = output["losses"]

        if self.trainer.use_fp16:
            self.trainer.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
        self.trainer.profile("Backward time")

    def _step(self, current_update):
        max_norm = self.trainer.running_config.max_norm
        if self.trainer.use_fp16:
            self.trainer.grad_scaler.unscale_(self.optimizer)
            self.trainer.profile("Unscale time")

        norm = clip_grad_norm(self.params, max_norm)
        self.trainer.profile("Clip grad time")

        if self.trainer.use_fp16:
            self.trainer.grad_scaler.step(self.optimizer)
            self.trainer.profile("Step time")
            self.trainer.grad_scaler.update()
            self.trainer.profile("Update time")
        else:
            self.optimizer.step()
            self.trainer.profile("Step time")

        if torch.isnan(norm).item() or torch.isinf(norm).item():
            self.num_skip += 1
            if self.num_skip >= 100:
                raise RuntimeError("Skipping iteration for more than 100 steps...")

            return current_update
        else:
            self.num_skip = 0

        if self.trainer.tb_writer is not None:
            self.trainer.tb_writer.add_scalars({"total_norm": norm}, current_update)

        current_update += 1

        return current_update

    def _sync_losses_and_metrics(self, split, output):
        losses = output["losses_stat"]
        metrics = output["metrics"]

        reduced_losses = reduce_dict(losses)
        reduced_metrics = reduce_dict(metrics)

        update_dict = {}
        update_dict.update(reduced_losses)
        update_dict.update(reduced_metrics)

        batch_size = self.trainer.running_config.batch_size
        self.trainer.meters[split].update(update_dict, batch_size)

    @torch.no_grad()
    def _update_info(self, split):
        current_update = self.trainer.current_update
        log_interval = self.trainer.log_interval

        if split == "train" and current_update % log_interval == 0:
            stats = {}
            ups = log_interval / self.trainer.timers["train"].unix_time_since_start()
            if "cuda" in str(self.trainer.device):
                stats["max mem"] = torch.cuda.max_memory_allocated() / 1000
                stats["max mem"] //= 1000

            stats.update(
                {
                    "epoch": self.current_epoch,
                    "data_epoch": self.trainer.current_epoch,
                    "update": current_update,
                    "max_update": self.trainer.max_update,
                    "lr": [
                        param_group["lr"] for param_group in self.optimizer.param_groups
                    ],
                    "ups": "{:.2f}".format(ups),
                    "time": self.trainer.timers["train"].get_time_since_start(),
                    "time_since_start": self.trainer.total_timer.get_time_since_start(),
                    "eta": self.trainer._calculate_time_left(),
                }
            )
            self.trainer._print_log(split, stats)
            self.trainer._update_tensorboard(split)
            self.trainer.timers["train"].reset()
        self.trainer.profile("Update info time")
