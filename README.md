# Similarity Learning with Hybrid CNN-Transformer Architectures in Medical Imaging

Reference architecture: `architecture.md`. Formulas: `math.md`. Positioning against the baseline: `novelty.md`.

## Setup

```bash
pip install -r requirements.txt
```

## Sanity check (no data, no GPU, no downloads required)

```bash
python scripts/smoke_test.py
```

Verifies tensor shapes end to end, checks gradients are finite, and confirms the set-aggregated
score is permutation-invariant to the order of the K synthetic draws (math.md §5).

## Data format

```
real_train.csv          columns: path,label      label=1 -> real_used, label=0 -> real_not_used
synth_train/              flat pool of GAN/diffusion-generated images
```

Same format for val/test/calibration splits. Keep calibration and test manifests strictly
disjoint from anything used in training (see `math.md` §9 on why conformal validity depends on this).

## Train

Single GPU:

```bash
python scripts/train.py \
  --train-manifest data/real_train.csv --train-synth-dir data/synth_train \
  --val-manifest data/real_val.csv --val-synth-dir data/synth_val \
  --batch-size 16 --k-draws 8 --amp-dtype bf16
```

Multi-GPU (DDP, single node):

```bash
bash scripts/launch_ddp.sh 4   # 4 GPUs
```

Key flags: `--grad-checkpointing` (trade compute for memory in the cross-attention stack),
`--torch-compile`, `--no-amp`, `--amp-dtype {bf16,fp16}`. All hyperparameters default from
`src/config.py`; anything not exposed as a flag can be edited there directly.

## Evaluate + calibrate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt \
  --manifest data/real_cal.csv --synth-dir data/synth_cal --out-npz cal_scores.npz

python scripts/evaluate.py --checkpoint checkpoints/best.pt \
  --manifest data/real_test.csv --synth-dir data/synth_test --out-npz test_scores.npz

python scripts/calibrate.py --cal-npz cal_scores.npz --test-npz test_scores.npz --alpha 0.1
```

`calibrate.py` reports empirical coverage on the test set against the target `1 - alpha`, and
how often the model abstains (prediction set size ≠ 1) — see `math.md` §9.

## Performance notes

- **AMP**: bf16 by default (no `GradScaler` needed on Ampere+ GPUs); fp16 falls back to
  `GradScaler` automatically.
- **DDP**: `scripts/train.py` reads `RANK`/`WORLD_SIZE`/`LOCAL_RANK` from `torchrun`; falls back
  to single-process on a plain `python scripts/train.py` call.
- **Class-balanced batches**: `src/sampler.py` guarantees every batch is 50/50 `real_used` /
  `real_not_used`, sharded correctly across DDP ranks — required for `SupConLoss` to have
  positive pairs at N~100/class (math.md §7).
- **Gradient checkpointing**: `--grad-checkpointing` recomputes the cross-attention stack's
  activations on the backward pass instead of storing them; use if `K` × batch size runs out of
  memory (architecture.md §4 on the B×K compute cost of cross-attention).
- **Gradient accumulation**: `cfg.grad_accum_steps` in `src/config.py` for effective batch sizes
  larger than what fits in memory at once.
