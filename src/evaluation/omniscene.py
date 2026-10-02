"""Shared mini/total/official-weight evaluation of the OmniScene protocol."""

import csv
import logging
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..dataset.shims.omniscene_shim import crop_batch, crop_views
from ..geometry.omniscene_alignment import align_input_cameras
from ..misc.omniscene_runtime import capture_rng, parameter_counts, restore_rng, write_json
from .metrics import compute_lpips, compute_pcc, compute_psnr, compute_ssim

LOG = logging.getLogger(__name__)
VIEW_GROUPS = {"all_18": slice(0, 18), "novel_12": slice(0, 12), "input_6": slice(12, 18)}
METRICS = ("psnr", "ssim", "lpips", "pcc")


@contextmanager
def evaluation_state(model):
    modes = [(module, module.training) for module in model.modules()]
    distill = model.encoder.distill
    rng = capture_rng()
    try:
        model.eval()
        model.encoder.distill = False
        with torch.no_grad():
            yield
    finally:
        model.encoder.distill = distill
        for module, training in modes:
            module.training = training
        restore_rng(rng)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def reconstruct_timed(model, cpu_images, device, patch_size=14, global_step=0):
    # This API receives CPU RGB directly; target data transfer is outside both timers.
    synchronize(device)
    with_transfer_start = time.perf_counter()
    images = cpu_images.to(device, non_blocking=True)
    synchronize(device)
    reconstruction_start = time.perf_counter()
    images = crop_views({"image": images}, patch_size)["image"]
    result = model.encoder(images, global_step=global_step, visualization_dump=None)
    synchronize(device)
    end = time.perf_counter()
    return result, {"reconstruction_ms": (end - reconstruction_start) * 1000,
                    "with_transfer_ms": (end - with_transfer_start) * 1000}


@torch.no_grad()
def score_groups(gt, prediction, gt_depth, pred_depth, groups, compute_depth=True):
    # Chunk LPIPS by view to bound evaluation memory independently of 18-view count.
    image_scores = {
        "psnr": compute_psnr(gt, prediction),
        "ssim": compute_ssim(gt, prediction),
        "lpips": torch.cat([compute_lpips(a[None], b[None]) for a, b in zip(gt, prediction)]),
    }
    result = {}
    for group in groups:
        indices = VIEW_GROUPS[group]
        result[group] = {name: values[indices].double().mean().item()
                         for name, values in image_scores.items()}
        if compute_depth:
            result[group]["pcc"] = compute_pcc(gt_depth[indices], pred_depth[indices]).item()
    return result


def summarize_records(records, groups):
    summary = {}
    for group in groups:
        rows = [row for row in records if row["view_group"] == group]
        entry = {"num_bins": len(rows), "undefined_counts": {}}
        for name in METRICS:
            values = [row[name] for row in rows if name in row]
            if not values:
                continue
            undefined = sum(not math.isfinite(x) for x in values)
            entry[name] = None if undefined else float(np.mean(values))
            entry["undefined_counts"][name] = undefined
        summary[group] = entry
    return summary


def timing_stats(values):
    if not values:
        return {"num_bins": 0, "mean_ms": None, "median_ms": None, "p95_ms": None}
    return {"num_bins": len(values), "mean_ms": float(np.mean(values)),
            "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def evaluate_omniscene(model, loader, test_cfg, output_dir, *, step=0, metadata=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    patch_size = loader.dataset.cfg.patch_size
    groups = test_cfg.view_groups
    expected = len(loader.dataset)
    records, timings, alignments = [], [], []
    processed = 0
    write_json(output_dir / "evaluation_summary.json", {"complete": False, "num_bins": 0, "expected_bins": expected})
    write_json(output_dir / "model_parameters.json", parameter_counts(model))
    if metadata is not None:
        write_json(output_dir / "run_metadata.json", metadata)
    fieldnames = ["bin_token", "view_group", *METRICS, "pcc_status", "alignment_fallback"]
    with (output_dir / "metrics_per_bin.csv").open("w", newline="") as file, evaluation_state(model):
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, cpu_batch in enumerate(loader):
            if test_cfg.limit_batches is not None and index >= test_cfg.limit_batches:
                break
            result, timing = reconstruct_timed(model, cpu_batch["context"]["image"], device, patch_size, step)
            batch = crop_batch(cpu_batch, patch_size)
            context, target = batch["context"], batch["target"]
            target_c2w, alignment = align_input_cameras(
                result.pred_context_pose["extrinsic"], context["extrinsics"].to(device), target["extrinsics"].to(device))
            token = batch["scene"][0]
            alignments.append({"bin_token": token, **alignment[0]})
            h, w = target["image"].shape[-2:]
            with torch.autocast(device_type=device.type, enabled=False):
                rendered = model.decoder.forward(
                    result.gaussians, target_c2w, target["intrinsics"].to(device),
                    torch.full((1, 18), 0.01, device=device), torch.full((1, 18), 100.0, device=device),
                    (h, w), "depth")
                if test_cfg.compute_scores:
                    scores = score_groups(
                        target["image"][0].to(device), rendered.color[0].float(),
                        target["rel_depth"][0].to(device) if test_cfg.compute_pcc else None,
                        rendered.depth[0].float(), groups, test_cfg.compute_pcc)
                    for group, values in scores.items():
                        pcc_status = "defined" if math.isfinite(values.get("pcc", math.nan)) else "undefined_or_disabled"
                        row = {"bin_token": token, "view_group": group, **values,
                               "pcc_status": pcc_status, "alignment_fallback": alignment[0]["fallback"]}
                        records.append(row)
                        writer.writerow(row)
                    file.flush()
            if test_cfg.save_image:
                folder = output_dir / "images" / token
                folder.mkdir(parents=True, exist_ok=True)
                for view, rgb in enumerate(rendered.color[0]):
                    image = (rgb.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    Image.fromarray(image).save(folder / f"{view:02d}.png")
            timings.append({"bin_token": token, "warmup": index < test_cfg.eval_time_skip_steps, **timing})
            processed += 1
            del result, rendered, batch, context, target, cpu_batch
            if processed % 100 == 0:
                LOG.info("OmniScene %s: %d/%d bins", loader.dataset.split, processed, expected)
    measured = [t for t in timings if not t["warmup"]]
    time_summary = {
        "reconstruction_ms_per_bin": timing_stats([t["reconstruction_ms"] for t in measured]),
        "reconstruction_with_transfer_ms_per_bin": timing_stats([t["with_transfer_ms"] for t in measured]),
        "warmup_bins": min(test_cfg.eval_time_skip_steps, processed),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "gpu_isolation": "not_enforced; record external contention when reporting performance",
        "boundary": "CPU RGB ready -> H2D -> GPU crop and complete student Gaussian reconstruction; synchronization at boundaries",
        "excluded": ["data_loading", "teacher", "camera_alignment", "rendering", "metrics", "file_output"],
        "per_bin": timings,
    }
    summary = {"complete": processed == expected, "num_bins": processed, "expected_bins": expected,
               "split": loader.dataset.split, "step": step, "debug_limit_batches": test_cfg.limit_batches,
               "loaded_resolution": loader.dataset.cfg.input_image_shape,
               "effective_resolution": [x // patch_size * patch_size for x in loader.dataset.cfg.input_image_shape],
               "pixel_protocol": "full_center_crop", "depth_mode": "RGB+D",
               "camera_alignment": "six_input_orientation_assisted_sim3", "groups": summarize_records(records, groups)}
    write_json(output_dir / "camera_alignment.json", alignments)
    write_json(output_dir / "reconstruction_time.json", time_summary)
    write_json(output_dir / "evaluation_summary.json", summary)
    LOG.info("Evaluation saved to %s: %s", output_dir, summary)
    return summary, time_summary
