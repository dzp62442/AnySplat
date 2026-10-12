"""Validation diagnostics; statistics always describe the unscaled depth values."""

from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps

from .image_io import save_image
from .omniscene_runtime import write_json


def view_statistics(values):
    rows = []
    for image in values.detach().float().cpu():
        rows.append({"min": image.min().item(), "max": image.max().item(),
                     "mean": image.mean().item(), "std": image.std(unbiased=False).item(),
                     "is_constant": bool((image == image.flatten()[0]).all()),
                     "all_zero": bool((image == 0).all()),
                     "finite_fraction": torch.isfinite(image).float().mean().item()})
    return rows


def save_reconstruction_diagnostics(folder, images, encoded, rendered):
    """All arrays are for batch element 0; depth colors share one per-sample range."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    head = encoded.depth_dict["depth"][0].squeeze(-1).detach().float().cpu()
    depth = rendered.depth[0].detach().float().cpu()
    alpha = rendered.alpha[0].detach().float().cpu()
    for name, value in (("depth_head", head), ("rendered_depth", depth), ("alpha", alpha)):
        np.save(folder / f"{name}.npy", value.numpy())
    joined = torch.cat((head, depth), dim=0)
    finite = joined[torch.isfinite(joined)]
    low, high = (finite.min().item(), finite.max().item()) if finite.numel() else (0., 0.)
    scaled = ((joined - low) / max(high - low, 1e-8)).nan_to_num().clamp(0, 1)
    colored = torch.from_numpy(colormaps["turbo"](scaled.numpy())[..., :3]).permute(0, 3, 1, 2)
    views = head.shape[0]
    mosaic = torch.cat([torch.cat(tuple(row), dim=-1) for row in colored.split(views)], dim=-2)
    save_image(mosaic, folder / "depth.png")
    save_image(torch.cat(tuple(alpha), dim=-1)[None].expand(3, -1, -1), folder / "alpha.png")
    rgb = torch.cat((torch.cat(tuple(images[0]), dim=-1),
                     torch.cat(tuple(rendered.color[0]), dim=-1)), dim=-2)
    save_image(rgb, folder / "rgb.png")
    cameras = {}
    for name in ("pred_context_pose", "construction_pose"):
        pose = getattr(encoded, name, None)
        if pose is not None:
            cameras[name] = {k: v[0].detach().float().cpu().tolist() for k, v in pose.items()}
    write_json(folder / "statistics.json", {
        "depth_head": view_statistics(head), "rendered_depth": view_statistics(depth),
        "alpha": view_statistics(alpha), "depth_color_range": [low, high],
        "depth_rows": ["depth_head", "rendered_depth"], "cameras": cameras,
    })
