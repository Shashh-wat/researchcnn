"""Run a trained checkpoint over a manifest, print metrics, and dump raw (score, label) pairs
for downstream conformal calibration (scripts/calibrate.py).

Run: python scripts/evaluate.py --checkpoint checkpoints/best.pt \
        --manifest real_test.csv --synth-dir synth_test/ \
        --out-npz test_scores.npz --out-metrics test_metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoint import load_checkpoint, load_config_from_checkpoint
from src.config import Config
from src.dataset import RealSyntheticSetDataset
from src.distributed import pick_device
from src.metrics import compute_metrics
from src.model import MembershipFingerprintNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--synth-dir", required=True)
    p.add_argument("--out-npz", default=None)
    p.add_argument("--out-metrics", default=None, help="Path to write metrics as JSON (default: <checkpoint dir>/eval_metrics.json)")
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()

    device = pick_device()
    # Rebuild the exact architecture the checkpoint was trained with (d_model, k_draws, ...)
    # rather than trusting CLI flags to match — see src/checkpoint.py docstring.
    cfg = load_config_from_checkpoint(args.checkpoint, Config, map_location=device)
    cfg.pretrained_backbones = False  # weights come from the checkpoint, not ImageNet
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    model = MembershipFingerprintNet(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    ds = RealSyntheticSetDataset(args.manifest, args.synth_dir, k=cfg.k_draws, image_size=cfg.image_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for real_img, synth_imgs, labels in loader:
            real_img, synth_imgs, labels = real_img.to(device), synth_imgs.to(device), labels.to(device)
            out = model(real_img, synth_imgs, labels)
            all_scores.append(out["score"].cpu())
            all_labels.append(labels.cpu())

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(scores, labels)
    print(metrics)

    if args.out_npz:
        np.savez(args.out_npz, scores=scores, labels=labels)
        print(f"saved raw scores/labels to {args.out_npz}")

    out_metrics_path = args.out_metrics or str(Path(args.checkpoint).parent / "eval_metrics.json")
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "synth_dir": args.synth_dir,
        "n_samples": int(len(labels)),
        "metrics": {k: (float(v) if not isinstance(v, dict) else v) for k, v in metrics.items()},
    }
    Path(out_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"saved metrics to {out_metrics_path}")


if __name__ == "__main__":
    main()
