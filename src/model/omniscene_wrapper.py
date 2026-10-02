"""AnySplat's original objectives, restricted to the six simultaneous RGB views."""

import logging

import torch
from einops import rearrange

from ..dataset.shims.omniscene_shim import crop_batch
from ..evaluation.metrics import compute_psnr
from ..misc.omniscene_runtime import capture_rng, restore_rng
from ..misc.image_io import save_image
from .model_wrapper import ModelWrapper

LOG = logging.getLogger(__name__)


class OmniSceneModelWrapper(ModelWrapper):
    def __init__(self, *args, patch_size=14, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size
        self.suppressed_updates = 0
        self._resume_rng = None

    def on_train_epoch_start(self):
        # The sampler uses virtual shuffled epochs and resumes by consumed steps.
        pass

    def on_validation_epoch_start(self):
        pass

    def _loss(self, batch, stage):
        batch = self.data_shim(crop_batch(batch, self.patch_size))
        context = batch["context"]
        if self.train_cfg.use_dynamic_mask:
            context["rgb_loss_mask"] = context["masks"]
        images = (context["image"] + 1) / 2
        encoded, rendered = self.model(images, self.global_step)
        batch["using_index"] = torch.arange(images.shape[1], device=images.device)
        depth = {**encoded.depth_dict, "distill_infos": encoded.distill_infos}
        total = images.new_zeros(())
        log_options = dict(on_step=stage == "train", on_epoch=stage == "val", batch_size=1)
        with torch.autocast(device_type=images.device.type, enabled=False):
            for loss_fn in self.losses:
                value = loss_fn(rendered, batch, encoded.gaussians, depth, self.global_step)
                self.log(f"{stage}/loss_{loss_fn.name}", value, **log_options)
                total = total + value
            if self.model.encoder.distill:
                values = self.loss_distill(encoded.distill_infos, encoded.pred_pose_enc_list, rendered, batch)
                total = total + values["loss_distill"]
                for key, value in values.items():
                    self.log(f"{stage}/{key}", value, **log_options)
        psnr = compute_psnr(rearrange(images, "b v c h w -> (b v) c h w"),
                            rearrange(rendered.color, "b v c h w -> (b v) c h w")).mean()
        self.log(f"{stage}/psnr_input", psnr, **log_options)
        self.log(f"{stage}/loss_total", total, **log_options)
        if stage == "val":
            comparison = torch.cat((torch.cat(tuple(images[0]), dim=-1),
                                    torch.cat(tuple(rendered.color[0]), dim=-1)), dim=-2)
            path = self.train_cfg.output_path / "val" / f"step-{self.global_step:08d}" / f"{batch['scene'][0]}.png"
            save_image(comparison, path)
        return total

    def training_step(self, batch, batch_idx):
        total = self._loss(batch, "train")
        self.log("info/global_step", float(self.global_step), on_step=True, batch_size=1)
        if self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            LOG.info("train step %d, bin=%s, loss=%.6f", self.global_step, batch["scene"], total.detach().item())
        # Retain the author's high-loss gradient suppression, not dataset filtering.
        if self.global_step > 1000 and total > 0.2:
            self.suppressed_updates += 1
            self.log("train/suppressed_updates", float(self.suppressed_updates), batch_size=1)
            total = total * 1e-10
        return total

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._loss(batch, "val")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["omniscene_model_cfg"] = self.checkpoint_model_cfg
        checkpoint["omniscene_protocol"] = getattr(self, "checkpoint_protocol", None)
        checkpoint["omniscene_rng"] = capture_rng()
        checkpoint["omniscene_suppressed_updates"] = self.suppressed_updates
        checkpoint["omniscene_consumed_samples"] = int(self.global_step)

    def on_load_checkpoint(self, checkpoint):
        self._resume_rng = checkpoint.get("omniscene_rng")
        self.suppressed_updates = checkpoint.get("omniscene_suppressed_updates", 0)

    def on_train_start(self):
        # Restore after setup/model construction and dataloader initialization.
        if self._resume_rng is not None:
            restore_rng(self._resume_rng)
            self._resume_rng = None
