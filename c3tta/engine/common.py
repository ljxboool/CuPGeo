from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
from contextlib import nullcontext
import numpy as np
import torch


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    if not enabled:
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    amp = getattr(torch, "amp", None)
    scaler_cls = getattr(amp, "GradScaler", None) if amp is not None else None
    if scaler_cls is not None:
        try:
            return scaler_cls(device.type, enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def load_trusted_checkpoint(path: str | os.PathLike[str], map_location: str | torch.device = "cpu"):
    """Load a checkpoint produced by this project, including optimizer/RNG state.

    PyTorch 2.6+ defaults to ``weights_only=True``. Our resumable checkpoints
    also contain NumPy/Python RNG state, so trusted project checkpoints require
    the legacy loader. Never use this helper for an untrusted downloaded file.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
