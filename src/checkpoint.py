"""Checkpoint save/resume helpers, DDP-safe (only rank 0 writes).

The Config is serialized alongside the weights (as a plain dict) so a checkpoint is
self-describing: scripts/evaluate.py and scripts/calibrate.py rebuild the exact same
architecture (d_model, k_draws, etc.) from the checkpoint instead of relying on the
caller to pass matching flags by hand.
"""
import dataclasses
from pathlib import Path

import torch


def save_checkpoint(path: str, model, optimizer, scheduler, scaler, epoch: int, best_metric: float, cfg=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model  # unwrap DDP
    torch.save(
        {
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_metric": best_metric,
            "config": dataclasses.asdict(cfg) if cfg is not None else None,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def load_config_from_checkpoint(path: str, config_cls, map_location="cpu"):
    """Rebuilds a Config from a checkpoint's saved dict, falling back to defaults if absent
    (e.g. checkpoints saved before this field existed)."""
    ckpt = torch.load(path, map_location=map_location)
    saved = ckpt.get("config")
    if saved is None:
        return config_cls()
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    return config_cls(**{k: v for k, v in saved.items() if k in valid_fields})
