"""Turns a flat pool of real images into the real_used/real_not_used split + manifest CSVs
that src/dataset.py and the training scripts expect (README.md data layout).

This does NOT touch synthetic images — there's no GAN yet. Run this now to lock in the
real_used/real_not_used split; train a GAN on the `real_used` folder later, then point
--train-synth-dir at whatever that GAN generates.

Run (e.g. on a folder of downloaded NIH ChestX-ray14 images):
    python scripts/prepare_manifest.py --images-dir data/nih_raw/images \
        --out-dir data --n-used 500 --n-not-used 500 --val-frac 0.15 --test-frac 0.15
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def write_manifest(path: Path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, help="Flat directory of source images")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--n-used", type=int, default=500, help="How many real images become real_used (label=1)")
    p.add_argument("--n-not-used", type=int, default=500, help="How many become real_not_used (label=0)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true", help="Copy files into --out-dir/real_used etc. instead of referencing originals in place")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    needed = args.n_used + args.n_not_used
    if len(all_images) < needed:
        raise ValueError(f"only {len(all_images)} images found in {images_dir}, need {needed}")

    rng = random.Random(args.seed)
    rng.shuffle(all_images)
    used_pool = all_images[: args.n_used]
    not_used_pool = all_images[args.n_used : args.n_used + args.n_not_used]

    if args.copy:
        used_dir = out_dir / "real_used"
        not_used_dir = out_dir / "real_not_used"
        used_dir.mkdir(exist_ok=True)
        not_used_dir.mkdir(exist_ok=True)
        used_pool = [Path(shutil.copy(p, used_dir / p.name)) for p in used_pool]
        not_used_pool = [Path(shutil.copy(p, not_used_dir / p.name)) for p in not_used_pool]

    def split(pool):
        n = len(pool)
        n_val = int(n * args.val_frac)
        n_test = int(n * args.test_frac)
        return pool[n_val + n_test:], pool[:n_val], pool[n_val:n_val + n_test]

    used_train, used_val, used_test = split(used_pool)
    not_used_train, not_used_val, not_used_test = split(not_used_pool)

    def rows(pool, label):
        return [(str(path.resolve()), label) for path in pool]

    write_manifest(out_dir / "real_train.csv", rows(used_train, 1) + rows(not_used_train, 0))
    write_manifest(out_dir / "real_val.csv", rows(used_val, 1) + rows(not_used_val, 0))
    write_manifest(out_dir / "real_test.csv", rows(used_test, 1) + rows(not_used_test, 0))

    print(f"real_used pool: {len(used_pool)} -> train/val/test = {len(used_train)}/{len(used_val)}/{len(used_test)}")
    print(f"real_not_used pool: {len(not_used_pool)} -> train/val/test = {len(not_used_train)}/{len(not_used_val)}/{len(not_used_test)}")
    print(f"wrote manifests to {out_dir}/real_{{train,val,test}}.csv")
    print()
    print("NEXT STEP: train a GAN on the images listed as label=1 in real_train.csv "
          "(the real_used split) to produce the synthetic pool — no synth images exist yet.")


if __name__ == "__main__":
    main()
