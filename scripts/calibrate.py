"""Split conformal calibration (math.md §9). Post-hoc, no retraining.

Run: python scripts/calibrate.py --cal-npz cal_scores.npz --test-npz test_scores.npz --alpha 0.1

`*.npz` files come from scripts/evaluate.py --out-npz on disjoint calibration/test manifests.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.conformal import SplitConformalCalibrator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cal-npz", required=True)
    p.add_argument("--test-npz", required=True)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--out-json", default="conformal_qhat.json")
    args = p.parse_args()

    cal = np.load(args.cal_npz)
    test = np.load(args.test_npz)

    calibrator = SplitConformalCalibrator(alpha=args.alpha)
    q_hat = calibrator.calibrate(cal["scores"], cal["labels"])
    print(f"q_hat = {q_hat:.4f} (target coverage {1 - args.alpha:.0%})")

    with open(args.out_json, "w") as f:
        json.dump({"alpha": args.alpha, "q_hat": q_hat}, f, indent=2)

    sets = calibrator.predict_set(test["scores"])
    labels = test["labels"]

    covered = sum(1 for s, y in zip(sets, labels) if y in s)
    abstained = sum(1 for s in sets if len(s) != 1)
    print(f"empirical test coverage: {covered / len(labels):.3f} (target {1 - args.alpha:.2f})")
    print(f"abstained (set size != 1): {abstained}/{len(labels)} ({abstained / len(labels):.1%})")


if __name__ == "__main__":
    main()
