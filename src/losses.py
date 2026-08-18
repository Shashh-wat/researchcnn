"""SupConLoss (math.md §7) and CombinedLoss (math.md §8)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive loss, single-view variant (Khosla et al. 2020, math.md §7)."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        b = embeddings.shape[0]
        z = F.normalize(embeddings, dim=-1)
        sim = (z @ z.t()) / self.temperature                      # (B,B)

        self_mask = torch.eye(b, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, float("-inf"))

        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)     # (B,B)
        pos_mask = label_eq & (~self_mask)

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)

        pos_counts = pos_mask.sum(dim=1)
        has_pos = pos_counts > 0
        if has_pos.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        masked_log_prob = log_prob.masked_fill(~pos_mask, 0.0)  # avoid 0 * -inf = nan on the diagonal
        mean_log_prob_pos = masked_log_prob.sum(dim=1)[has_pos] / pos_counts[has_pos]
        return -mean_log_prob_pos.mean()


class CombinedLoss(nn.Module):
    """L = lambda1 * BCE + lambda2 * ArcFace + lambda3 * SupCon (math.md §8)."""

    def __init__(self, lambda_bce: float, lambda_arcface: float, lambda_supcon: float, supcon_temperature: float):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_arcface = lambda_arcface
        self.lambda_supcon = lambda_supcon
        self.supcon = SupConLoss(temperature=supcon_temperature)

    def forward(self, outputs: dict, labels: torch.Tensor) -> dict:
        bce = F.binary_cross_entropy(outputs["score"], labels.float())
        arcface = F.cross_entropy(outputs["arcface_logits"], labels)
        supcon = self.supcon(outputs["identity_embedding"], labels)
        total = self.lambda_bce * bce + self.lambda_arcface * arcface + self.lambda_supcon * supcon
        return {"total": total, "bce": bce, "arcface": arcface, "supcon": supcon}
