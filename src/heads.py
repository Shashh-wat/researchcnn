"""RelationHead (math.md §4) and ArcFaceHead (math.md §6)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationHead(nn.Module):
    """u_k = [h_r-h_g, h_r*h_g, h_r, h_g] -> r_k (for set aggregation) and z_k (auxiliary logit)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.to_relation = nn.Sequential(nn.Linear(4 * d_model, d_model), nn.ReLU())
        self.aux_logit = nn.Sequential(
            nn.Linear(4 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, h_r: torch.Tensor, h_g: torch.Tensor):
        u = torch.cat([h_r - h_g, h_r * h_g, h_r, h_g], dim=-1)
        r_k = self.to_relation(u)
        z_k = self.aux_logit(u)
        return r_k, z_k


class ArcFaceHead(nn.Module):
    """Additive angular margin head (Deng et al. 2019), math.md §6."""

    def __init__(self, d_model: int, num_classes: int, scale: float = 16.0, margin: float = 0.3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.scale = scale
        self.margin = margin
        self.eps = 1e-7

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        e_hat = F.normalize(embedding, dim=-1)
        w_hat = F.normalize(self.weight, dim=-1)
        cosine = e_hat @ w_hat.t()                              # (B, C)
        theta = torch.acos(torch.clamp(cosine, -1 + self.eps, 1 - self.eps))
        target_logit = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).float()
        logits = cosine * (1 - one_hot) + target_logit * one_hot
        return logits * self.scale
