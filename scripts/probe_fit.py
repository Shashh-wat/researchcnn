"""Memory probe: exit 0 if (batch, k) runs two training steps with headroom on this GPU, 3 if it does not.

Usage: python scripts/probe_fit.py BATCH K
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.losses import CombinedLoss
from src.model import MembershipFingerprintNet

# Neighbours on a shared GPU can grow, so the probe's peak must stay well under what was free at the start.
HEADROOM = 0.85


def main() -> int:
    b, k = int(sys.argv[1]), int(sys.argv[2])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    free = torch.cuda.mem_get_info()[0] if dev.type == "cuda" else None
    cfg = Config(pretrained_backbones=False, use_grad_checkpointing=True, k_draws=k)
    try:
        model = MembershipFingerprintNet(cfg).to(dev).train()
        loss_fn = CombinedLoss(cfg.lambda_bce, cfg.lambda_arcface, cfg.lambda_supcon, cfg.supcon_temperature)
        labels = torch.tensor([0, 1] * (b // 2), device=dev)
        x = torch.randn(b, 3, 224, 224, device=dev)
        y = torch.randn(b, k, 3, 224, 224, device=dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(2):
            loss = loss_fn(model(x, y, labels), labels)["total"]
            opt.zero_grad()
            loss.backward()
            opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"batch={b} K={k}: out of memory")
        return 3
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated()
        ok = peak <= HEADROOM * free
        print(f"batch={b} K={k}: peak {peak / 1e9:.1f} GB of {free / 1e9:.1f} GB free -> {'fits' if ok else 'too tight'}")
        return 0 if ok else 3
    print(f"batch={b} K={k}: ran on CPU (no memory check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
