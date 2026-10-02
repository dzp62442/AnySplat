"""Center cropping with complete normalized-intrinsics updates."""


def crop_views(views: dict, patch_size: int = 14) -> dict:
    h, w = views["image"].shape[-2:]
    new_h, new_w = h // patch_size * patch_size, w // patch_size * patch_size
    top, left = (h - new_h) // 2, (w - new_w) // 2
    result = dict(views)
    for key in ("image", "masks", "valid_mask", "rel_depth"):
        if key in views:
            result[key] = views[key][..., top:top + new_h, left:left + new_w].contiguous()
    if "intrinsics" in views:
        k = views["intrinsics"].clone()
        k[..., 0, :] *= w / new_w
        k[..., 1, :] *= h / new_h
        k[..., 0, 2] -= left / new_w
        k[..., 1, 2] -= top / new_h
        result["intrinsics"] = k
    return result


def crop_batch(batch: dict, patch_size: int = 14) -> dict:
    return {key: crop_views(value, patch_size) if key in ("context", "target") else value
            for key, value in batch.items()}
