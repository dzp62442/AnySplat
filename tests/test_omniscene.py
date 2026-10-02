"""Protocol regression tests; all temporary files belong under /tmp."""

import copy
import itertools
import json
import math
import pickle
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch import Callback, LightningDataModule, Trainer, seed_everything
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics import PearsonCorrCoef

from src.config import load_typed_root_config
from src.dataset.dataset_omniscene import CAMERAS, DatasetOmniScene, DatasetOmniSceneCfg
from src.dataset.omniscene_data_module import ResumableShuffleSampler
from src.dataset.shims.omniscene_shim import crop_batch, crop_views
from src.evaluation.metrics import compute_pcc
from src.evaluation.omniscene import evaluation_state, score_groups, summarize_records
from src.geometry.omniscene_alignment import align_input_cameras
from src.global_cfg import set_cfg
from src.loss.loss_distill import DistillLoss
from src.misc.omniscene_callbacks import OmniSceneProgress
from src.misc.omniscene_runtime import AtomicCheckpointIO, link_checkpoint, parameter_counts, resolve_resume
from src.model.omniscene_wrapper import OmniSceneModelWrapper

ROOT = Path(__file__).resolve().parents[1]


def configuration(resolution="112x200"):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        raw = compose(config_name="main", overrides=[f"+experiment=omniscene_{resolution}"])
    return load_typed_root_config(raw), raw


def make_data(root, *, targets=True):
    version = root / "interp_12Hz_trainval"
    (version / "bin_infos_3.2m").mkdir(parents=True)
    for split in ("train", "val"):
        (version / f"bins_{split}_3.2m.json").write_text(json.dumps({"bins": ["bin0"]}))
    sensors = {}
    for camera, name in enumerate(CAMERAS):
        sensors[name] = []
        for view in range(3 if targets else 1):
            stem = f"{camera}-{view}"
            c2w = np.eye(4)
            c2w[0, 3] = camera * 10 + view
            sensors[name].append({"data_path": f"/datasets/nuScenes/samples/{stem}.jpg",
                                  "sensor2lidar_transform": c2w})
            for kind in ("small", "param_small", "mask_small", "dpt_small"):
                (root / f"samples_{kind}").mkdir(exist_ok=True)
            Image.fromarray(np.full((224, 400, 3), camera * 30 + view, np.uint8)).save(root / "samples_small" / f"{stem}.jpg")
            if targets:
                (root / "samples_param_small" / f"{stem}.json").write_text(json.dumps({
                    "camera_intrinsic": [[300, 0, 180], [0, 310, 105], [0, 0, 1]]}))
                np.save(root / "samples_dpt_small" / f"{stem}.npy", np.linspace(1, 5, 224 * 400, dtype=np.float32).reshape(224, 400))
                # Deliberately omit input mask assets: center masks must be all one.
                if view:
                    mask = np.zeros((224, 400), np.uint8)
                    mask[:, 200:] = 255
                    Image.fromarray(mask).save(root / "samples_mask_small" / f"{stem}.png")
    with (version / "bin_infos_3.2m" / "bin0.pkl").open("wb") as f:
        pickle.dump({"sensor_info": sensors}, f)  # No LIDAR_TOP, intentionally.


class ProtocolTests(unittest.TestCase):
    def test_two_resolved_configurations(self):
        for resolution, shape in (("112x200", [112, 200]), ("224x400", [224, 400])):
            cfg, raw = configuration(resolution)
            self.assertEqual(cfg.dataset[0].omniscene.input_image_shape, shape)
            self.assertEqual(cfg.trainer.max_steps, 100001)
            self.assertEqual(cfg.trainer.val_check_interval, 1000)
            self.assertEqual(cfg.train.weight_depth, 0)
            self.assertTrue(cfg.model.encoder.distill)
            self.assertEqual(raw.loss.depth_consis.weight, 0.1)
            self.assertEqual([cfg.data_loader.train.batch_size, cfg.data_loader.val.batch_size, cfg.data_loader.test.batch_size], [1, 1, 1])

    def test_training_requires_only_six_rgb_files(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            make_data(root, targets=False)
            cfg = DatasetOmniSceneCfg("omniscene", [root], [112, 200])
            sample = DatasetOmniScene(cfg, "train")[0]
            self.assertNotIn("target", sample)
            self.assertNotIn("intrinsics", sample["context"])
            self.assertNotIn("extrinsics", sample["context"])
            self.assertNotIn("rel_depth", sample["context"])
            self.assertEqual(tuple(sample["context"]["image"].shape), (6, 3, 112, 200))
            self.assertTrue(sample["context"]["masks"].eq(1).all())
            self.assertFalse(sample["context"]["valid_mask"].any())

    def test_target_order_and_existing_mask_semantics(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            make_data(root)
            cfg = DatasetOmniSceneCfg("omniscene", [root], [112, 200])
            sample = DatasetOmniScene(cfg, "test")[0]
            x = sample["target"]["extrinsics"][:, 0, 3].tolist()
            self.assertEqual(x, [camera * 10 + view for camera in range(6) for view in (1, 2)] + [camera * 10 for camera in range(6)])
            self.assertTrue(sample["target"]["masks"][12:].eq(1).all())
            self.assertLess(sample["target"]["masks"][:12].mean(), 1)
            self.assertEqual(tuple(sample["target"]["rel_depth"].shape), (18, 112, 200))
            self.assertTrue(torch.equal(sample["context"]["image"], sample["target"]["image"][12:]))

    def test_explicit_mini_total_splits(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            base = root / "interp_12Hz_trainval"
            base.mkdir()
            tokens = [str(i) for i in range(30080)]
            (base / "bins_val_3.2m.json").write_text(json.dumps({"bins": tokens}))
            cfg = DatasetOmniSceneCfg("omniscene", [root], [112, 200])
            self.assertEqual(DatasetOmniScene(cfg, "test", split="total").bin_tokens, tokens)
            self.assertEqual(DatasetOmniScene(cfg, "test", split="mini").bin_tokens, tokens[::14][:2048])
            self.assertEqual(DatasetOmniScene(cfg, "val").bin_tokens, tokens[:30000:3000][:10])

    def test_crop_preserves_rays_with_off_center_principal_point(self):
        for h, w, new_w in ((112, 200, 196), (224, 400, 392)):
            k = torch.tensor([[0.75, 0.0, 0.43], [0, 1.1, 0.47], [0, 0, 1.0]])[None, None]
            ramp = torch.arange(h * w).reshape(1, 1, h, w)
            views = {"image": ramp.unsqueeze(2).expand(-1, -1, 3, -1, -1), "intrinsics": k,
                     "masks": ramp, "rel_depth": ramp}
            cropped = crop_views(views)
            self.assertEqual(cropped["image"].shape[-2:], (h, new_w))
            self.assertTrue(torch.equal(cropped["masks"], cropped["rel_depth"]))
            left = (w - new_w) // 2
            before = ((left + 17) / w - k[0, 0, 0, 2]) / k[0, 0, 0, 0]
            after = (17 / new_w - cropped["intrinsics"][0, 0, 0, 2]) / cropped["intrinsics"][0, 0, 0, 0]
            torch.testing.assert_close(before, after)
            torch.testing.assert_close(k[0, 0, 0, 2], torch.tensor(0.43))

    def test_six_camera_sim3_maps_all_targets(self):
        gen = torch.Generator().manual_seed(1)
        pred = torch.eye(4).repeat(1, 6, 1, 1).double()
        pred[0, :, :3, 3] = torch.randn(6, 3, generator=gen).double()
        rotation = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float64)
        translation = torch.tensor([2, -4, 1], dtype=torch.float64)
        known = pred.clone()
        known[..., :3, :3] = rotation @ pred[..., :3, :3]
        known[..., :3, 3] = 3 * (pred[..., :3, 3] @ rotation.T) + translation
        target = known.repeat(1, 3, 1, 1)
        actual, record = align_input_cameras(pred, known, target)
        torch.testing.assert_close(actual, pred.repeat(1, 3, 1, 1))
        self.assertAlmostEqual(record[0]["scale"], 3)
        self.assertIsNone(record[0]["fallback"])

    def test_degenerate_prediction_is_retained(self):
        pred = torch.eye(4).repeat(1, 6, 1, 1)
        known = pred.clone()
        known[0, :, 0, 3] = torch.arange(6)
        result, record = align_input_cameras(pred, known, known.repeat(1, 3, 1, 1))
        self.assertEqual(result.shape[1], 18)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(record[0]["scale"], 1)

    def test_pcc_matches_reference_without_cross_call_state(self):
        gen = torch.Generator().manual_seed(3)
        gt, pred = torch.randn(18, 5, 7, generator=gen), torch.randn(18, 5, 7, generator=gen)
        for indices in (slice(None), slice(0, 12), slice(12, 18), slice(None)):
            reference = PearsonCorrCoef()(gt[indices].reshape(-1), pred[indices].reshape(-1))
            torch.testing.assert_close(compute_pcc(gt[indices], pred[indices]), reference)

    def test_pcc_is_group_flattened_not_mean_view_correlation(self):
        gt = torch.arange(18 * 12).reshape(18, 3, 4).float()
        pred = gt + torch.arange(18).reshape(18, 1, 1).remainder(2) * 500
        images = torch.zeros(18, 3, 3, 4)
        with patch("src.evaluation.omniscene.compute_ssim", return_value=torch.zeros(18)), \
             patch("src.evaluation.omniscene.compute_lpips", return_value=torch.zeros(1)):
            scores = score_groups(images, images + 0.1, gt, pred, ["all_18", "novel_12"])
        self.assertAlmostEqual(scores["all_18"]["pcc"], compute_pcc(gt, pred).item())
        self.assertLess(scores["all_18"]["pcc"], 0.9)  # Each per-view correlation would be 1.

    def test_undefined_pcc_is_not_silently_dropped(self):
        records = [{"view_group": "all_18", "pcc": 1.0}, {"view_group": "all_18", "pcc": float("nan")}]
        summary = summarize_records(records, ["all_18"])["all_18"]
        self.assertIsNone(summary["pcc"])
        self.assertEqual(summary["undefined_counts"]["pcc"], 1)
        self.assertEqual(summary["num_bins"], 2)

    def test_zero_normal_weight_does_not_require_external_intrinsics(self):
        depth = torch.ones(1, 6, 3, 4)
        pose = torch.zeros(1, 6, 9, requires_grad=True)
        loss = DistillLoss(weight_pose=10, weight_depth=0, weight_normal=0)
        result = loss({"pred_pose_enc_list": [torch.ones_like(pose)], "depth_map": depth[..., None],
                       "conf_mask": depth.bool()}, [pose], SimpleNamespace(depth=depth),
                      {"context": {"valid_mask": torch.zeros_like(depth, dtype=torch.bool)}})
        result["loss_distill"].backward()
        self.assertGreater(pose.grad.abs().sum(), 0)

    def test_eval_restores_distill_modes_rng_and_trainability(self):
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(2, 2)
        model.encoder.distill = True
        model.encoder.eval()
        state = torch.get_rng_state()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with evaluation_state(model):
                self.assertFalse(model.encoder.distill)
                torch.rand(5)
                raise RuntimeError("intentional")
        self.assertTrue(model.training)
        self.assertFalse(model.encoder.training)
        self.assertTrue(model.encoder.distill)
        self.assertTrue(model.encoder.weight.requires_grad)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_sampler_resumes_consumed_position_across_virtual_epochs(self):
        reference = list(itertools.islice(iter(ResumableShuffleSampler(7, 12)), 30))
        resumed = list(itertools.islice(iter(ResumableShuffleSampler(7, 12, 11)), 19))
        self.assertEqual(reference[11:], resumed)
        self.assertEqual(sorted(reference[:7]), list(range(7)))

    def test_checkpoint_selection_uses_state_step_and_recovers_broken_last(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            root = Path(tmp)
            io = AtomicCheckpointIO()
            for filename, step in (("step-9999.ckpt", 3), ("step-0001.ckpt", 9)):
                io.save_checkpoint({"global_step": step, "optimizer_states": [], "loops": {}}, root / filename)
            (root / "last.ckpt").write_bytes(b"interrupted")
            path, state = resolve_resume(root, None, True)
            self.assertEqual(state["global_step"], 9)
            link_checkpoint(path, root / "last.ckpt")
            link_checkpoint(path, root / "last.ckpt")
            self.assertFalse((root / "last.ckpt.link.tmp").exists())
            self.assertEqual(io.load_checkpoint(root / "last.ckpt")["global_step"], 9)


class ToyDataset(Dataset):
    def __len__(self):
        return 7

    def __getitem__(self, index):
        return {"x": torch.tensor([index / 7 + 1.0]), "scene": str(index)}


class ToyData(LightningDataModule):
    def __init__(self, start):
        super().__init__()
        self.start = start

    def train_dataloader(self):
        return DataLoader(ToyDataset(), batch_size=1, sampler=ResumableShuffleSampler(7, 22, self.start),
                          generator=torch.Generator().manual_seed(11))

    def val_dataloader(self):
        return DataLoader(ToyDataset(), batch_size=1, generator=torch.Generator().manual_seed(15))

    def mini_dataloader(self):
        return None


class ToyWrapper(OmniSceneModelWrapper):
    def __init__(self, cfg):
        model = torch.nn.Module()
        model.encoder = torch.nn.Linear(1, 1)
        model.encoder.pred_pose = False
        model.encoder.distill = False
        model.decoder = torch.nn.Identity()
        super().__init__(cfg.optimizer, cfg.test, cfg.train, model, [], None)
        self.checkpoint_model_cfg = {}
        self.seen = []

    def _loss(self, batch, stage):
        if stage == "train":
            self.seen.append(batch["scene"][0])
            noise = torch.rand_like(batch["x"]) + random.random() + np.random.rand()
        else:
            noise = torch.zeros_like(batch["x"])
        return (self.model.encoder(batch["x"]) - noise).square().mean()


class StopAfter(Callback):
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if trainer.global_step == 3:
            trainer.should_stop = True


class ResumeTests(unittest.TestCase):
    def test_lightning_resume_weights_scheduler_rng_and_sample_sequence(self):
        cfg, raw = configuration()
        raw.trainer.max_steps = 6
        raw.optimizer.warm_up_steps = 2
        set_cfg(raw)
        cfg.trainer.max_steps = 6
        cfg.optimizer.warm_up_steps = 2
        cfg.notifications.feishu_enabled = False
        cfg.trainer.val_check_interval = 2
        cfg.train.eval_model_every_n_val = 1
        cfg.checkpointing.every_n_train_steps = 2
        mini_steps = []

        def fake_evaluate(model, loader, test_cfg, output_dir, *, step, metadata):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            mini_steps.append(step)
            values = {k: 0.5 for k in ("psnr", "ssim", "lpips", "pcc")}
            return ({"complete": True, "num_bins": 1, "expected_bins": 1,
                     "groups": {"all_18": values, "novel_12": values}},
                    {"reconstruction_ms_per_bin": {"mean_ms": 1},
                     "reconstruction_with_transfer_ms_per_bin": {"mean_ms": 2}})

        def run(root, start=0, checkpoint=None, stop=False):
            seed_everything(42)
            root.mkdir(exist_ok=True)
            (root / "resolved_config.yaml").write_text("test: true\n")
            model = ToyWrapper(cfg)
            progress = OmniSceneProgress(cfg, root, {"loaded_resolution": [112, 200],
                                                    "effective_resolution": [112, 196], "weights": "toy"}, start)
            callbacks = [progress, StopAfter()] if stop else [progress]
            trainer = Trainer(accelerator="cpu", devices=1, max_steps=6, max_epochs=-1,
                              logger=False, enable_checkpointing=False, enable_progress_bar=False,
                              enable_model_summary=False, callbacks=callbacks, plugins=[AtomicCheckpointIO()],
                              val_check_interval=2, check_val_every_n_epoch=None, num_sanity_val_steps=0,
                              default_root_dir=str(root))
            trainer.fit(model, datamodule=ToyData(start), ckpt_path=str(checkpoint) if checkpoint else None)
            path = progress.save(trainer)
            if not stop:
                progress.finish(trainer, model)
            return model, trainer, path

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp, patch("src.misc.omniscene_callbacks.evaluate_omniscene", side_effect=fake_evaluate):
            root = Path(tmp)
            whole, whole_trainer, _ = run(root / "whole")
            first, _, saved = run(root / "resumed", stop=True)
            after, after_trainer, _ = run(root / "resumed", start=3, checkpoint=saved)
            self.assertEqual(first.seen + after.seen, whole.seen)
            for key, value in whole.model.state_dict().items():
                torch.testing.assert_close(value, after.model.state_dict()[key], atol=0, rtol=0)
            self.assertEqual(after_trainer.global_step, 6)
            self.assertEqual(after_trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                             whole_trainer.lr_scheduler_configs[0].scheduler.state_dict())
            state = json.loads((root / "resumed" / "run_state.json").read_text())
            self.assertTrue(state["final_mini_complete"])
            self.assertEqual(mini_steps, [2, 4, 6, 6, 2, 4, 6, 6])


if __name__ == "__main__":
    unittest.main()
