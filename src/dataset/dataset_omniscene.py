"""OmniScene's existing six-center / twelve-novel camera protocol.

Training intentionally does not read novel images, calibration or depth labels.
The manifest is trusted: no scene filtering or dataset quality audit takes place.
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


@dataclass
class DatasetOmniSceneCfg:
    name: Literal["omniscene"]
    roots: list[Path]
    input_image_shape: list[int]
    data_version: str = "interp_12Hz_trainval"
    dataset_prefix: str = "/datasets/nuScenes"
    num_context_views: int = 6
    load_metric_depth: bool = False
    load_dynamic_mask: bool = True
    patch_size: int = 14


@dataclass
class DatasetOmniSceneCfgWrapper:
    omniscene: DatasetOmniSceneCfg


def asset_path(path: Path, kind: str, suffix: str) -> Path:
    """Change the camera asset directory, not arbitrary root-name substrings."""
    replacements = {"samples": f"samples_{kind}", "sweeps": f"sweeps_{kind}"}
    return Path(*(replacements.get(part, part) for part in path.parts)).with_suffix(suffix)


def relative_depth(disp: np.ndarray) -> np.ndarray:
    # Keep the SVF-GS / DepthSplat disparity conversion and operation order.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = min(disp.max() / (disp.min() + 0.001), 50.0)
        depth = 1.0 / np.maximum(disp, disp.max() / ratio)
        return (depth - depth.min()) / (depth.max() - depth.min())


class DatasetOmniScene(Dataset):
    def __init__(self, cfg: DatasetOmniSceneCfg, stage: str, view_sampler=None,
                 split: str = "total", compute_pcc: bool = True):
        self.cfg, self.stage, self.split = cfg, stage, split
        self.root = Path(cfg.roots[0])
        base = self.root / cfg.data_version
        filename = "bins_train_3.2m.json" if stage == "train" else "bins_val_3.2m.json"
        with (base / filename).open() as f:
            self.bin_tokens = json.load(f)["bins"]
        if stage == "val":
            self.bin_tokens = self.bin_tokens[:30000:3000][:10]
        elif stage == "test" and split == "mini":
            self.bin_tokens = self.bin_tokens[::14][:2048]
        self.load_rel_depth = stage == "test" and compute_pcc

    def __len__(self):
        return len(self.bin_tokens)

    def _views(self, infos, *, input_view: bool, calibration: bool):
        height, width = self.cfg.input_image_shape
        images, masks, intrinsics, extrinsics, depths = [], [], [], [], []
        for info in infos:
            path = Path(info["data_path"].replace(self.cfg.dataset_prefix, str(self.root)))
            if not path.is_absolute() and not path.is_relative_to(self.root):
                path = self.root / path
            with Image.open(asset_path(path, "small", ".jpg")) as source:
                original_width, original_height = source.size
                image = source.convert("RGB").resize((width, height))
                images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255)
            if calibration:
                with asset_path(path, "param_small", ".json").open() as f:
                    k = np.asarray(json.load(f)["camera_intrinsic"], dtype=np.float32)
                k[0] *= width / original_width
                k[1] *= height / original_height
                k[0] /= width
                k[1] /= height
                intrinsics.append(torch.from_numpy(k))
                extrinsics.append(torch.as_tensor(np.array(info["sensor2lidar_transform"], copy=True), dtype=torch.float32))
            if input_view or not self.cfg.load_dynamic_mask:
                masks.append(torch.ones(height, width))
            else:
                with Image.open(asset_path(path, "mask_small", ".png")) as mask:
                    mask = mask.convert("L").resize((width, height), Image.Resampling.BILINEAR)
                    masks.append(torch.from_numpy(np.array(mask)).float() / 255)
            if self.load_rel_depth:
                disp = np.load(asset_path(path, "dpt_small", ".npy")).astype(np.float32)
                if (original_height, original_width) != (height, width):
                    disp = np.array(Image.fromarray(disp).resize((width, height), Image.Resampling.BILINEAR))
                depths.append(torch.from_numpy(relative_depth(disp)))
        views = {"image": torch.stack(images), "masks": torch.stack(masks),
                 "index": torch.arange(len(infos))}
        if calibration:
            views.update(intrinsics=torch.stack(intrinsics), extrinsics=torch.stack(extrinsics))
        if depths:
            views["rel_depth"] = torch.stack(depths)
        return views

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        with (self.root / self.cfg.data_version / "bin_infos_3.2m" / f"{token}.pkl").open("rb") as f:
            info = pickle.load(f)["sensor_info"]
        context = self._views([info[cam][0] for cam in CAMERAS],
                              input_view=True, calibration=self.stage == "test")
        # This is the original loss API's "no external validity mask" sentinel.
        # It must NOT be replaced with the all-one dynamic RGB mask.
        context["valid_mask"] = torch.zeros_like(context["masks"], dtype=torch.bool)
        result = {"context": context, "scene": str(token)}
        if self.stage == "test":
            novel = self._views([info[cam][j] for cam in CAMERAS for j in (1, 2)],
                                input_view=False, calibration=True)
            result["target"] = {
                key: torch.cat((value, context[key]), dim=0)
                for key, value in novel.items() if key != "index"
            }
            result["target"]["index"] = torch.arange(18)
        return result
