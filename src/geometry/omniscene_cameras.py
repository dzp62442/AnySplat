"""A fixed camera gauge determined only by the six input calibrations."""

import torch


CAMERA_NORMALIZATION = "first_input_mean_center_radius_v1"


def apply_camera_normalization(c2w, transform):
    # Lightning wraps the full forward in autocast; calibration stays full precision.
    with torch.autocast(device_type=c2w.device.type, enabled=False):
        reference = transform["reference_c2w"]
        inverse_rotation = reference[:, :3, :3].transpose(-1, -2)
        result = c2w.clone()
        result[..., :3, :3] = inverse_rotation[:, None] @ c2w[..., :3, :3]
        delta = c2w[..., :3, 3] - reference[:, None, :3, 3]
        result[..., :3, 3] = (inverse_rotation[:, None] @ delta[..., None]).squeeze(-1)
        result[..., :3, 3] /= transform["scale"][:, None, None]
    return result


def normalize_input_cameras(c2w):
    centers = c2w[..., :3, 3]
    radius = (centers - centers.mean(dim=1, keepdim=True)).norm(dim=-1).mean(dim=1)
    # Numeric guard only; never reject/filter a dataset sample.
    transform = {"reference_c2w": c2w[:, 0].clone(), "scale": radius.clamp_min(1e-8)}
    return apply_camera_normalization(c2w, transform), transform


def pixel_intrinsics(normalized_intrinsics, image_shape):
    h, w = image_shape
    result = normalized_intrinsics.clone()
    result[..., 0, :] *= w
    result[..., 1, :] *= h
    return result
