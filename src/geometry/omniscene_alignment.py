"""One global Sim(3) from the six input cameras; no target-image optimization."""

import torch


@torch.no_grad()
def align_input_cameras(predicted_c2w, known_c2w, target_c2w):
    aligned, records = [], []
    for pred, known, target in zip(predicted_c2w.double(), known_c2w.double(), target_c2w.double()):
        reason = None
        rotation = torch.eye(3, device=pred.device, dtype=pred.dtype)
        scale = pred.new_tensor(1.0)
        translation = pred.new_zeros(3)
        if torch.isfinite(pred).all():
            relative = known[:, :3, :3] @ pred[:, :3, :3].transpose(-1, -2)
            u, _, vh = torch.linalg.svd(relative.sum(0))
            correction = torch.eye(3, device=pred.device, dtype=pred.dtype)
            correction[2, 2] = torch.linalg.det(u @ vh)
            rotation = u @ correction @ vh
            p, q = pred[:, :3, 3], known[:, :3, 3]
            p_centered, q_centered = p - p.mean(0), q - q.mean(0)
            denominator = p_centered.square().sum()
            if denominator > torch.finfo(pred.dtype).eps:
                fitted_scale = (q_centered * (p_centered @ rotation.T)).sum() / denominator
                if torch.isfinite(fitted_scale) and fitted_scale > torch.finfo(pred.dtype).eps:
                    scale = fitted_scale
                else:
                    reason = "nonpositive_scale_used_unit_scale"
            else:
                reason = "coincident_predicted_centers_used_unit_scale"
            translation = q.mean(0) - scale * (rotation @ p.mean(0))
        else:
            reason = "nonfinite_predicted_camera_used_identity"
        result = target.clone()
        result[:, :3, :3] = rotation.T @ target[:, :3, :3]
        result[:, :3, 3] = ((target[:, :3, 3] - translation) @ rotation) / scale
        aligned.append(result.to(target_c2w.dtype))
        records.append({"scale": scale.item(), "rotation": rotation.cpu().tolist(),
                        "translation": translation.cpu().tolist(), "fallback": reason})
    return torch.stack(aligned), records
