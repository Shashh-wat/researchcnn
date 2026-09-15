"""Materializes a small MedMNIST subset to a flat PNG folder, for the mini-demo pipeline
(see README.md "Mini demo" section). Uses PneumoniaMNIST at 64x64 (chest X-rays, matches
the project's medical-imaging domain) purely because it downloads instantly with no
registration wall — not chosen for any property of the pneumonia task itself, which is
irrelevant here; only the raw images are used, labels are discarded.

Run: python scripts/build_mini_demo_pool.py --out-dir mini_demo/pool --n-images 200
"""
import argparse
from pathlib import Path

import numpy as np
from medmnist import PneumoniaMNIST
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-images", type=int, default=200)
    p.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PneumoniaMNIST(split="train", download=True, size=args.size)
    n = min(args.n_images, len(ds))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=n, replace=False)

    for i, j in enumerate(idx):
        img, _ = ds[int(j)]  # PIL image, label discarded — only raw pixels matter here
        img.convert("RGB").save(out_dir / f"img_{i:05d}.png")

    print(f"wrote {n} images to {out_dir}")


if __name__ == "__main__":
    main()
