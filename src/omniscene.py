"""Hydra execution path for the approved OmniScene experiment protocol."""

import hashlib
import json
import logging
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_model

from .config import ModelCfg, load_typed_config
from .dataset.omniscene_data_module import OmniSceneDataModule
from .evaluation.omniscene import evaluate_omniscene
from .loss import get_losses
from .misc.omniscene_callbacks import OmniSceneProgress
from .misc.omniscene_runtime import AtomicCheckpointIO, resolve_resume, write_json
from .model.model import get_model
from .model.omniscene_wrapper import OmniSceneModelWrapper

LOG = logging.getLogger(__name__)


def model_config(value):
    return load_typed_config(DictConfig(value), ModelCfg, {tuple[float, float, float]: tuple})


def load_evaluation_model(cfg, checkpoint_state=None):
    if cfg.test.weights_source == "author":
        directory = Path(cfg.test.pretrained_path)
        with (directory / "config.json").open() as f:
            raw = json.load(f)
        model_cfg = model_config({"encoder": raw["encoder_cfg"], "decoder": raw["decoder_cfg"]})
        model_cfg.encoder = replace(model_cfg.encoder, initialize_from_vggt=False, distill=False, verbose=False)
        model = get_model(model_cfg.encoder, model_cfg.decoder)
        load_model(model, str(directory / "model.safetensors"), strict=True)
        source = str(directory.resolve())
        step = 0
    else:
        if checkpoint_state is None:
            if not cfg.checkpointing.load:
                raise ValueError("Training-model evaluation requires checkpointing.load=<complete .ckpt>")
            checkpoint_state = torch.load(cfg.checkpointing.load, map_location="cpu", mmap=True, weights_only=False)
        model_cfg = model_config(checkpoint_state.get("omniscene_model_cfg", asdict(cfg.model)))
        model_cfg.encoder = replace(model_cfg.encoder, initialize_from_vggt=False, distill=False, verbose=False)
        model = get_model(model_cfg.encoder, model_cfg.decoder)
        # Only the known model prefix and teacher are removed, never arbitrary missing keys.
        student = {key.removeprefix("model."): value for key, value in checkpoint_state["state_dict"].items()
                   if key.startswith("model.") and not key.startswith("model.encoder.distill_")}
        model.load_state_dict(student, strict=True)
        source = str(Path(cfg.checkpointing.load).resolve())
        step = int(checkpoint_state["global_step"])
    return model, source, step


def implementation_metadata():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted([*root.glob("src/**/*.py"), *root.glob("config/**/*.yaml")]):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"git_commit": sha, "implementation_sha256": digest.hexdigest()}


def run_omniscene(cfg, cfg_dict, output_dir):
    dataset_cfg = cfg.dataset[0].omniscene
    # These are execution/configuration contracts, not assertions about dataset quality.
    if len(cfg.dataset) != 1 or dataset_cfg.num_context_views != 6:
        raise ValueError("OmniScene uses one dataset and six center input views")
    if any(getattr(cfg.data_loader, stage).batch_size != 1 for stage in ("train", "val", "test")):
        raise ValueError("OmniScene train/val/test batch_size must all be 1")
    if cfg.trainer.devices != 1 or cfg.trainer.num_nodes != 1 or cfg.trainer.accumulate_grad_batches != 1:
        raise ValueError("The approved batch=1 protocol uses one GPU, one node, no accumulation")
    if dataset_cfg.load_metric_depth or cfg.train.supervision_views != "context":
        raise ValueError("This experiment uses RGB-only reconstruction and context-only supervision")
    if cfg.test.align_pose or cfg.test.camera_alignment != "input_sim3":
        raise ValueError("Use input_sim3; test-image pose optimization is disabled")
    if not {"all_18", "novel_12"}.issubset(cfg.test.view_groups) or not set(cfg.test.view_groups).issubset({"all_18", "novel_12", "input_6"}):
        raise ValueError("Evaluation must include all_18 and novel_12; input_6 is optional")
    if cfg.wandb["mode"] not in ("offline", "disabled"):
        raise ValueError("OmniScene logging must be offline or disabled")
    if not isinstance(cfg.trainer.val_check_interval, int) or cfg.trainer.val_check_interval <= 0:
        raise ValueError("Use a positive integer validation interval in optimizer steps")
    if cfg.checkpointing.save_weights_only:
        raise ValueError("Automatic training resume requires complete checkpoints")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed, workers=True)
    checkpoint_path, checkpoint_state = (None, None)
    if cfg.mode == "train":
        checkpoint_path, checkpoint_state = resolve_resume(output_dir / "checkpoints", cfg.checkpointing.load, cfg.checkpointing.auto_resume)
    protocol = {"dataset_root": str(Path(dataset_cfg.roots[0]).resolve()),
                "data_version": dataset_cfg.data_version,
                "input_image_shape": dataset_cfg.input_image_shape, "patch_size": dataset_cfg.patch_size,
                "num_context_views": 6, "supervision_views": "context"}
    if checkpoint_state is not None and checkpoint_state.get("omniscene_protocol") not in (None, protocol):
        raise ValueError("Checkpoint belongs to a different OmniScene experiment; use its matching work directory/configuration")
    start_step = int(checkpoint_state["global_step"]) if checkpoint_state is not None else 0
    state_path = output_dir / "run_state.json"
    if cfg.mode == "train" and start_step >= cfg.trainer.max_steps and state_path.is_file():
        with state_path.open() as f:
            previous = json.load(f)
        if previous.get("final_mini_complete") and previous.get("step") == start_step:
            LOG.info("Training and final mini already complete at step %d in %s", start_step, output_dir)
            return
    cfg.train.output_path = output_dir
    cfg_dict.train.output_path = str(output_dir)
    cfg_dict.test.output_path = str(output_dir)
    OmegaConf.save(cfg_dict, output_dir / "resolved_config.yaml", resolve=True)
    metadata = {**implementation_metadata(),
                "experiment": cfg.wandb["name"], "mode": cfg.mode,
                "loaded_resolution": dataset_cfg.input_image_shape,
                "effective_resolution": [x // dataset_cfg.patch_size * dataset_cfg.patch_size for x in dataset_cfg.input_image_shape],
                "dataset_root": str(dataset_cfg.roots[0]), "input_views": 6,
                "supervision_views": "context", "camera_alignment": cfg.test.camera_alignment,
                "precision": cfg.trainer.precision}
    data_module = OmniSceneDataModule(dataset_cfg, cfg.data_loader, cfg.test, start_step)
    if cfg.mode == "test":
        model, source, step = load_evaluation_model(cfg)
        metadata.update(weights=source, weights_source=cfg.test.weights_source, step=step,
                        loaded_model_config={"encoder": asdict(model.encoder.cfg), "decoder": asdict(model.decoder.cfg)})
        write_json(output_dir / "run_metadata.json", metadata)
        model.to(torch.device("cuda", 0))
        evaluate_omniscene(model, data_module.test_dataloader(), cfg.test, output_dir, step=step, metadata=metadata)
        return
    if cfg.checkpointing.init_source != "vggt":
        raise ValueError("Training initialization must be vggt; author weights are an evaluation mode")
    encoder_cfg = replace(cfg.model.encoder, initialize_from_vggt=checkpoint_path is None)
    model = get_model(encoder_cfg, cfg.model.decoder)
    wrapper = OmniSceneModelWrapper(cfg.optimizer, cfg.test, cfg.train, model, get_losses(cfg.loss), None,
                                   patch_size=dataset_cfg.patch_size)
    wrapper.checkpoint_model_cfg = asdict(cfg.model)
    wrapper.checkpoint_protocol = protocol
    metadata.update(weights=str(checkpoint_path) if checkpoint_path else encoder_cfg.vggt_pretrained_path,
                    weights_source="resumed_training" if checkpoint_path else "vggt_initialization",
                    loaded_model_config={"encoder": asdict(encoder_cfg), "decoder": asdict(cfg.model.decoder)})
    write_json(output_dir / "run_metadata.json", metadata)
    logger = (WandbLogger(project=cfg.wandb["project"], name=cfg.wandb["name"], mode="offline",
                          save_dir=str(output_dir), log_model=False, config=OmegaConf.to_container(cfg_dict, resolve=True))
              if cfg.wandb["mode"] == "offline" else CSVLogger(str(output_dir), name="logs"))
    progress = OmniSceneProgress(cfg, output_dir, metadata, start_step)
    trainer = Trainer(
        accelerator="gpu", devices=1, num_nodes=1, strategy="auto", max_epochs=-1,
        max_steps=cfg.trainer.max_steps, val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None, num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        precision=cfg.trainer.precision, accumulate_grad_batches=1,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        logger=logger, callbacks=[progress], enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        plugins=[AtomicCheckpointIO()], default_root_dir=str(output_dir),
        log_every_n_steps=cfg.train.print_log_every_n_steps,
    )
    del checkpoint_state  # Lightning restores via mmap; do not retain a second state mapping.
    trainer.fit(wrapper, datamodule=data_module, ckpt_path=str(checkpoint_path) if checkpoint_path else None)
    if trainer.global_step >= cfg.trainer.max_steps:
        progress.finish(trainer, wrapper)
    else:
        progress.save(trainer)
        LOG.info("Stopped at step %d; a resumable checkpoint was retained", trainer.global_step)
