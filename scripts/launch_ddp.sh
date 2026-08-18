#!/usr/bin/env bash
# Multi-GPU launch example (single node, N GPUs). Adjust --nproc_per_node to GPU count.
set -euo pipefail

NPROC=${1:-4}

torchrun --standalone --nproc_per_node="${NPROC}" scripts/train.py \
  --train-manifest data/real_train.csv \
  --train-synth-dir data/synth_train \
  --val-manifest data/real_val.csv \
  --val-synth-dir data/synth_val \
  --batch-size 8 \
  --k-draws 8 \
  --amp-dtype bf16 \
  --grad-checkpointing \
  --ckpt-dir checkpoints
