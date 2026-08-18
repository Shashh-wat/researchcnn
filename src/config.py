from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # embedding / attention
    d_model: int = 256
    n_heads: int = 8
    n_cross_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    use_grad_checkpointing: bool = False  # trade compute for memory in cross-attention stack

    # backbones
    vit_name: str = "vit_base_patch16_224"
    vit_embed_dim: int = 768
    resnet_out_channels: int = 2048
    pretrained_backbones: bool = True

    # token grid sizes (must match image_size / patch_size)
    image_size: int = 224
    resnet_grid: int = 7          # 224 / 32
    vit_patch: int = 16
    freq_patch: int = 16

    # set aggregation
    k_draws: int = 8

    # ArcFace
    arcface_scale: float = 16.0
    arcface_margin: float = 0.3
    num_classes: int = 2

    # SupCon
    supcon_temperature: float = 0.1

    # loss weights
    lambda_bce: float = 1.0
    lambda_arcface: float = 0.5
    lambda_supcon: float = 0.3

    # training
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8            # per-GPU batch size under DDP
    epochs: int = 30
    num_workers: int = 4
    seed: int = 42
    grad_accum_steps: int = 1
    warmup_steps: int = 500
    grad_clip_norm: float = 1.0
    val_every: int = 1             # epochs
    log_every: int = 20            # steps
    ckpt_dir: str = "checkpoints"
    resume_from: Optional[str] = None

    # performance
    use_amp: bool = True
    amp_dtype: str = "bf16"        # "bf16" or "fp16"
    use_torch_compile: bool = False
    channels_last: bool = True

    # conformal
    conformal_alpha: float = 0.1  # target miscoverage epsilon -> 90% coverage
