"""Local artifacts, checkpoint I/O, parameter counts and bounded notifications."""

import json
import logging
import math
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO

LOG = logging.getLogger(__name__)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(clean_json(value), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    os.replace(temporary, path)


class AtomicCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)

    def load_checkpoint(self, path, map_location=None, weights_only=False):
        return torch.load(str(path), map_location=map_location or "cpu", mmap=True, weights_only=False)


def link_checkpoint(source: Path, destination: Path):
    """Atomic hard link: last/final do not duplicate a multi-GB checkpoint."""
    temporary = destination.with_name(destination.name + ".link.tmp")
    temporary.unlink(missing_ok=True)
    os.link(source, temporary)
    os.replace(temporary, destination)
    # POSIX rename is a no-op when both names already point at the same inode.
    temporary.unlink(missing_ok=True)


def resolve_resume(directory: Path, explicit: str | None, auto_resume: bool):
    if explicit:
        candidates = [Path(explicit)]
    elif auto_resume:
        candidates = sorted(directory.glob("*.ckpt"), key=lambda p: (p.name != "last.ckpt", p.name))
    else:
        return None, None
    latest_path, latest_state, latest_step = None, None, -1
    seen = set()
    for path in candidates:
        try:
            identity = (path.stat().st_dev, path.stat().st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            state = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
            if any(key not in state for key in ("global_step", "optimizer_states", "loops")):
                raise ValueError("This is not a complete Lightning training checkpoint")
            step = int(state["global_step"])
            if step > latest_step:
                latest_path, latest_state, latest_step = path, state, step
        except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            if explicit:
                raise
            LOG.warning("Cannot restore checkpoint %s: %s; trying another saved checkpoint", path, exc)
    if candidates and latest_path is None:
        raise RuntimeError(f"No complete checkpoint can be restored from {directory}; existing files were preserved")
    return latest_path, latest_state


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def parameter_counts(model):
    counts = {key: {"trainable": 0, "frozen": 0, "total": 0}
              for key in ("reconstruction", "teacher", "training_model")}
    seen = set()
    for name, parameter in model.named_parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        group = "teacher" if "distill_" in name else "reconstruction"
        for key in (group, "training_model"):
            counts[key]["trainable" if parameter.requires_grad else "frozen"] += parameter.numel()
            counts[key]["total"] += parameter.numel()
    counts["definition"] = "Unique parameters under experiment freeze policy; evaluation updates none. Loss/metric networks excluded."
    return counts


def notify_feishu(cfg, subject, content):
    if not cfg.feishu_enabled:
        return
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from auto_monitor.send_feishu import send_feishu; "
        "send_feishu(sys.argv[2], sys.argv[3])"
    )
    try:
        subprocess.run([sys.executable, "-B", "-c", script, str(cfg.library_root), subject, content],
                       timeout=cfg.timeout_seconds, check=True, capture_output=True, text=True)
    except (subprocess.SubprocessError, OSError) as exc:
        # Do not print captured service output, which may contain credentials.
        LOG.warning("Feishu notification failed (%s); local results retained", type(exc).__name__)
