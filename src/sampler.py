"""Class-balanced batch sampler, DDP-aware.

SupCon needs both classes present in every batch to have positive pairs (math.md §7).
At N~100/class this can't be left to chance with plain shuffling, so batches are
constructed as (batch_size/2 real_used, batch_size/2 real_not_used), classes cycled
independently (with wraparound) since the two classes are rarely equal-sized. Balanced
batches are built once per epoch and then sharded across DDP ranks so every rank sees a
distinct, batch-aligned slice.
"""
import math

import torch
from torch.utils.data import Sampler


class ClassBalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size: int, num_replicas: int = 1, rank: int = 0, seed: int = 0):
        assert batch_size % 2 == 0, "batch_size must be even for exact 50/50 class balance"
        self.labels = list(labels)
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.idx_by_class = {
            c: [i for i, y in enumerate(self.labels) if y == c] for c in set(self.labels)
        }
        assert len(self.idx_by_class) == 2, "expected exactly 2 classes"

        n_total = len(self.labels)
        self._num_batches = max(1, n_total // batch_size)
        # make num_batches divisible by num_replicas so every rank gets an equal share
        self._num_batches -= self._num_batches % self.num_replicas
        self._num_batches = max(self.num_replicas, self._num_batches)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return (self._num_batches // self.num_replicas) * self.batch_size

    def _cycled_shuffled(self, indices, g, length):
        indices = indices.copy()
        out = []
        while len(out) < length:
            perm = torch.randperm(len(indices), generator=g).tolist()
            out.extend(indices[i] for i in perm)
        return out[:length]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        classes = sorted(self.idx_by_class.keys())
        needed = self._num_batches * self.half
        pool_a = self._cycled_shuffled(self.idx_by_class[classes[0]], g, needed)
        pool_b = self._cycled_shuffled(self.idx_by_class[classes[1]], g, needed)

        batches = []
        for i in range(self._num_batches):
            batch = (
                pool_a[i * self.half:(i + 1) * self.half]
                + pool_b[i * self.half:(i + 1) * self.half]
            )
            perm = torch.randperm(len(batch), generator=g).tolist()
            batches.append([batch[j] for j in perm])

        my_batches = batches[self.rank::self.num_replicas]
        flat = [idx for batch in my_batches for idx in batch]
        return iter(flat)
