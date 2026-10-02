"""Checkpoint cadence and the shared periodic/final mini evaluation."""

import json
import logging
import time
from pathlib import Path

from lightning.pytorch import Callback

from ..evaluation.omniscene import evaluate_omniscene
from .omniscene_runtime import link_checkpoint, notify_feishu, parameter_counts, write_json

LOG = logging.getLogger(__name__)


class OmniSceneProgress(Callback):
    def __init__(self, cfg, output_dir, metadata, start_step=0):
        self.cfg, self.output_dir, self.metadata = cfg, Path(output_dir), metadata
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_step = start_step
        self.last_mini_step = -1
        self.last_saved_step = -1
        self.started = time.monotonic()

    def state_dict(self):
        return {"last_mini_step": self.last_mini_step}

    def load_state_dict(self, state):
        self.last_mini_step = state.get("last_mini_step", -1)

    def on_fit_start(self, trainer, pl_module):
        state_path = self.output_dir / "run_state.json"
        if state_path.is_file():
            with state_path.open() as f:
                state = json.load(f)
            # A sidecar can be newer than the checkpoint, but never reuse a future evaluation.
            side_step = state.get("last_mini_step", -1)
            if side_step <= self.start_step:
                self.last_mini_step = max(self.last_mini_step, side_step)
        counts = parameter_counts(pl_module.model)
        write_json(self.output_dir / "model_parameters.json", counts)
        notify_feishu(self.cfg.notifications, "AnySplat OmniScene 训练启动",
                      f"实验：{self.cfg.wandb['name']}\n工作目录：{self.output_dir}\n"
                      f"恢复步数：{self.start_step} / {self.cfg.trainer.max_steps}\n"
                      f"加载尺寸：{self.metadata['loaded_resolution']}\n有效尺寸：{self.metadata['effective_resolution']}\n"
                      f"初始化/恢复：{self.metadata['weights']}\n模型参数：{counts}")

    def save(self, trainer, final=False):
        step = int(trainer.global_step)
        path = self.checkpoint_dir / f"step-{step:08d}.ckpt"
        if self.last_saved_step != step or not path.is_file():
            trainer.save_checkpoint(path, weights_only=False)
            self.last_saved_step = step
        if self.cfg.checkpointing.save_last:
            link_checkpoint(path, self.checkpoint_dir / "last.ckpt")
        if final:
            link_checkpoint(path, self.checkpoint_dir / "final.ckpt")
        keep = self.cfg.checkpointing.save_top_k
        if keep > 0:
            for old in sorted(self.checkpoint_dir.glob("step-*.ckpt"))[:-keep]:
                old.unlink()
        return path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        interval = self.cfg.checkpointing.every_n_train_steps
        if interval > 0 and trainer.global_step % interval == 0:
            self.save(trainer)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        interval = self.cfg.train.eval_model_every_n_val
        period = int(self.cfg.trainer.val_check_interval) * interval
        step = int(trainer.global_step)
        if interval > 0 and step > 0 and step % period == 0 and step != self.last_mini_step:
            self.run_mini(trainer, pl_module, final=False)

    def run_mini(self, trainer, pl_module, *, final):
        step = int(trainer.global_step)
        if not final:
            self.save(trainer)
        path = self.output_dir / "eval" / f"{'final-mini' if final else 'mini'}-step-{step:08d}"
        checkpoint = self.checkpoint_dir / ("final.ckpt" if final else f"step-{step:08d}.ckpt")
        summary, timing = evaluate_omniscene(
            pl_module.model, trainer.datamodule.mini_dataloader(), self.cfg.test, path,
            step=step, metadata={**self.metadata, "checkpoint": str(checkpoint), "step": step})
        (path / "resolved_config.yaml").write_text((self.output_dir / "resolved_config.yaml").read_text())
        if summary["complete"]:
            self.last_mini_step = step
        write_json(self.output_dir / "run_state.json", {
            "step": step, "last_mini_step": self.last_mini_step,
            "training_complete": step >= self.cfg.trainer.max_steps,
            "final_mini_complete": final and summary["complete"],
            "final_mini_path": str(path) if final else None,
        })
        logs = {}
        lines = [f"实验：{self.cfg.wandb['name']}", f"step: {step}",
                 f"mini bins: {summary['num_bins']} / {summary['expected_bins']}", f"结果：{path}",
                 f"checkpoint：{checkpoint}"]
        for group in ("all_18", "novel_12"):
            values = summary["groups"][group]
            fields = []
            for name in ("psnr", "ssim", "lpips", "pcc"):
                value = values.get(name)
                fields.append(f"{name}={value:.{3 if name == 'psnr' else 4}f}" if value is not None else f"{name}=undefined")
                if value is not None:
                    logs[f"mini/{group}/{name}"] = value
            lines.append(group + ": " + ", ".join(fields))
        lines.append(f"完整重建 ms/bin（不含 H2D、对齐、渲染和指标）：{timing['reconstruction_ms_per_bin']['mean_ms']}")
        lines.append(f"含 H2D 的入口 ms/bin：{timing['reconstruction_with_transfer_ms_per_bin']['mean_ms']}")
        updates = step - self.start_step
        if updates > 0:
            remaining = (time.monotonic() - self.started) / updates * max(0, self.cfg.trainer.max_steps - step)
            lines.append(f"预计剩余小时：{remaining / 3600:.2f}")
        if trainer.logger:
            trainer.logger.log_metrics(logs, step=step)
        LOG.info("%s", "\n".join(lines))
        notify_feishu(self.cfg.notifications, "AnySplat 最终 mini 完成" if final else "AnySplat mini 完成", "\n".join(lines))

    def finish(self, trainer, pl_module):
        self.save(trainer, final=True)
        if self.cfg.train.final_mini_test:
            # Lightning teardown moves the model/optimizer to CPU after fit().
            # Only the student is needed for final evaluation; leave the teacher on CPU.
            device = trainer.strategy.root_device
            for name, module in pl_module.model.encoder.named_children():
                if not name.startswith("distill_"):
                    module.to(device)
            # Encoder-level positional/camera tokens are registered in its child modules.
            pl_module.model.decoder.to(device)
            self.run_mini(trainer, pl_module, final=True)
