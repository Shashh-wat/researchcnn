"""MembershipFingerprintNet: full assembly, see architecture.md §1 for the data-flow diagram."""
import torch
import torch.nn as nn

from src.attention import CrossAttentionStack, PMAPool
from src.backbone import Tokenizer
from src.heads import ArcFaceHead, RelationHead


class MembershipFingerprintNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = Tokenizer(cfg)  # shared weights, real & synthetic (Siamese)

        self.cross_attn = CrossAttentionStack(
            cfg.d_model, cfg.n_heads, cfg.n_cross_layers, cfg.ffn_mult, cfg.dropout,
            use_grad_checkpointing=cfg.use_grad_checkpointing,
        )
        self.pool_real = PMAPool(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.pool_synth = PMAPool(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.pool_set = PMAPool(cfg.d_model, cfg.n_heads, cfg.dropout)

        self.relation_head = RelationHead(cfg.d_model)
        self.score_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(), nn.Linear(cfg.d_model, 1)
        )
        self.arcface_head = ArcFaceHead(
            cfg.d_model, cfg.num_classes, cfg.arcface_scale, cfg.arcface_margin
        )

    def forward(self, real_img: torch.Tensor, synth_imgs: torch.Tensor, labels: torch.Tensor = None) -> dict:
        """
        real_img:   (B, 3, H, W)
        synth_imgs: (B, K, 3, H, W)
        labels:     (B,) int64 in {0,1}, required to compute arcface_logits (teacher-forced margin)
        """
        b, k = synth_imgs.shape[0], synth_imgs.shape[1]

        t_r = self.tokenizer(real_img)                                  # (B, 441, d)
        t_g = self.tokenizer(synth_imgs.reshape(b * k, *synth_imgs.shape[2:]))  # (B*K, 441, d)

        t_r_rep = t_r.unsqueeze(1).expand(b, k, *t_r.shape[1:]).reshape(b * k, *t_r.shape[1:])

        attended_r, attended_g = self.cross_attn(t_r_rep, t_g)          # (B*K, 441, d) each

        h_r = self.pool_real(attended_r).view(b, k, -1)                 # (B, K, d)
        h_g = self.pool_synth(attended_g).view(b, k, -1)                # (B, K, d)

        r_k, z_k = self.relation_head(h_r, h_g)                         # (B,K,d), (B,K,1)

        pooled_rel = self.pool_set(r_k)                                  # (B, d)
        score = torch.sigmoid(self.score_head(pooled_rel)).squeeze(-1)   # (B,)

        identity_embedding = h_r.mean(dim=1)                              # (B, d)

        out = {
            "score": score,
            "identity_embedding": identity_embedding,
            "per_draw_logit": z_k.squeeze(-1),  # (B, K) auxiliary, inspection only
        }
        if labels is not None:
            out["arcface_logits"] = self.arcface_head(identity_embedding, labels)
        return out
