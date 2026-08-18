"""Shape/gradient/permutation-invariance sanity check. No data or network downloads required.

Run: python scripts/smoke_test.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.losses import CombinedLoss
from src.model import MembershipFingerprintNet


def main():
    torch.manual_seed(0)
    cfg = Config(pretrained_backbones=False, k_draws=4, d_model=64, n_heads=4, n_cross_layers=1)

    model = MembershipFingerprintNet(cfg)
    loss_fn = CombinedLoss(cfg.lambda_bce, cfg.lambda_arcface, cfg.lambda_supcon, cfg.supcon_temperature)

    b, k = 3, cfg.k_draws
    real_img = torch.randn(b, 3, cfg.image_size, cfg.image_size)
    synth_imgs = torch.randn(b, k, 3, cfg.image_size, cfg.image_size)
    labels = torch.tensor([0, 1, 1], dtype=torch.long)

    print(f"tokenizer output tokens per image: {model.tokenizer.num_tokens} (expect 49+196+196=441)")
    assert model.tokenizer.num_tokens == 441

    out = model(real_img, synth_imgs, labels)
    print("score:", out["score"].shape, out["score"].detach())
    print("identity_embedding:", out["identity_embedding"].shape)
    print("arcface_logits:", out["arcface_logits"].shape)
    print("per_draw_logit:", out["per_draw_logit"].shape)

    assert out["score"].shape == (b,)
    assert out["identity_embedding"].shape == (b, cfg.d_model)
    assert out["arcface_logits"].shape == (b, cfg.num_classes)
    assert out["per_draw_logit"].shape == (b, k)
    assert torch.all((out["score"] >= 0) & (out["score"] <= 1))

    losses = loss_fn(out, labels)
    print("losses:", {k_: round(v.item(), 4) for k_, v in losses.items()})
    losses["total"].backward()

    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    n_params_with_grad = len(grad_norms)
    n_params_total = sum(1 for _ in model.parameters())
    print(f"params with nonzero grad path: {n_params_with_grad}/{n_params_total}")
    assert n_params_with_grad > 0
    assert all(g == g for g in grad_norms), "NaN gradient detected"

    # Permutation invariance of the set-aggregated score w.r.t. order of the K synthetic draws.
    model.eval()
    with torch.no_grad():
        perm = torch.randperm(k)
        out_a = model(real_img, synth_imgs, labels)
        out_b = model(real_img, synth_imgs[:, perm], labels)
        diff = (out_a["score"] - out_b["score"]).abs().max().item()
    print(f"max |score(order A) - score(order B)| over K permutation: {diff:.2e}")
    assert diff < 1e-4, "set-aggregated score is not permutation-invariant"

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
