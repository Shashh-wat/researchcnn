"""Bidirectional cross-attention (math.md §2) and PMA set-pooling (math.md §3, §5)."""
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class _CrossBlock(nn.Module):
    """One direction, one layer: X attends to Y, then FFN. Pre-LN residual transformer block."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        kv = self.norm_kv(y)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionStack(nn.Module):
    """L stacked bidirectional cross-attention blocks between real and synthetic token bags."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ffn_mult: int = 4,
                 dropout: float = 0.1, use_grad_checkpointing: bool = False):
        super().__init__()
        self.r_to_g = nn.ModuleList(
            [_CrossBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.g_to_r = nn.ModuleList(
            [_CrossBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.use_grad_checkpointing = use_grad_checkpointing

    def _run(self, layer: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_grad_checkpointing and self.training:
            return cp.checkpoint(layer, x, y, use_reentrant=False)
        return layer(x, y)

    def forward(self, t_r: torch.Tensor, t_g: torch.Tensor):
        x_r, x_g = t_r, t_g
        for layer_r, layer_g in zip(self.r_to_g, self.g_to_r):
            new_r = self._run(layer_r, x_r, x_g)
            new_g = self._run(layer_g, x_g, x_r)
            x_r, x_g = new_r, new_g
        return x_r, x_g  # attended_r, attended_g


class PMAPool(nn.Module):
    """Pooling by Multihead Attention (Set Transformer, Lee et al. 2019). Permutation-invariant over N."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d) -> (B, d), invariant to permutations along N
        b = x.shape[0]
        seed = self.seed.expand(b, -1, -1)
        pooled, _ = self.attn(seed, x, x, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        return pooled
