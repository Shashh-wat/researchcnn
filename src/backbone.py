"""Token extraction: spatial-CNN, spatial-ViT, and frequency streams (math.md §1)."""
import timm
import torch
import torch.nn as nn
import torchvision


class SpatialCNNTokenizer(nn.Module):
    """ResNet-50 up to Layer4, kept as a 7x7 spatial map -> 49 tokens. No GAP."""

    def __init__(self, d_model: int, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = torchvision.models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.proj = nn.Linear(2048, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)                       # (B, 2048, 7, 7)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)   # (B, 49, 2048)
        return self.proj(tokens)                   # (B, 49, d)


class SpatialViTTokenizer(nn.Module):
    """ViT-B/16 patch tokens (CLS dropped) -> 196 tokens."""

    def __init__(self, d_model: int, vit_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        self.num_prefix_tokens = getattr(self.vit, "num_prefix_tokens", 1)
        self.proj = nn.Linear(self.vit.embed_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)                  # (B, 1+196, embed_dim) typically
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]    # drop CLS/register tokens
        return self.proj(patch_tokens)                          # (B, 196, d)


class FrequencyTokenizer(nn.Module):
    """Log-magnitude 2D-FFT spectrum, patch-embedded on a 14x14 grid -> 196 tokens."""

    def __init__(self, d_model: int, patch: int = 16):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch, stride=patch)

    @staticmethod
    def _to_log_spectrum(x: torch.Tensor) -> torch.Tensor:
        # x: (B,3,H,W) in [0,1]-ish (already normalized upstream) -> grayscale
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        log_mag = torch.log1p(spectrum.abs())
        mean = log_mag.mean(dim=(-2, -1), keepdim=True)
        std = log_mag.std(dim=(-2, -1), keepdim=True) + 1e-6
        return (log_mag - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = self._to_log_spectrum(x)            # (B,1,H,W)
        tokens = self.patch_embed(spec)             # (B, d, 14, 14)
        b, d, h, w = tokens.shape
        return tokens.flatten(2).transpose(1, 2)     # (B, 196, d)


class TokenFusion(nn.Module):
    """Adds modality + positional embeddings to each stream and concatenates them."""

    def __init__(self, d_model: int, n_res: int, n_vit: int, n_freq: int):
        super().__init__()
        self.n_res, self.n_vit, self.n_freq = n_res, n_vit, n_freq
        self.modality_embed = nn.Embedding(3, d_model)  # 0=res, 1=vit, 2=freq
        self.pos_res = nn.Parameter(torch.randn(1, n_res, d_model) * 0.02)
        self.pos_vit = nn.Parameter(torch.randn(1, n_vit, d_model) * 0.02)
        self.pos_freq = nn.Parameter(torch.randn(1, n_freq, d_model) * 0.02)

    def forward(self, t_res: torch.Tensor, t_vit: torch.Tensor, t_freq: torch.Tensor) -> torch.Tensor:
        t_res = t_res + self.pos_res + self.modality_embed.weight[0]
        t_vit = t_vit + self.pos_vit + self.modality_embed.weight[1]
        t_freq = t_freq + self.pos_freq + self.modality_embed.weight[2]
        return torch.cat([t_res, t_vit, t_freq], dim=1)  # (B, 441, d)


class Tokenizer(nn.Module):
    """Full image -> token-bag pipeline (math.md §1). Shared weights across real/synthetic (Siamese)."""

    def __init__(self, cfg):
        super().__init__()
        self.cnn = SpatialCNNTokenizer(cfg.d_model, pretrained=cfg.pretrained_backbones)
        self.vit = SpatialViTTokenizer(cfg.d_model, cfg.vit_name, pretrained=cfg.pretrained_backbones)
        self.freq = FrequencyTokenizer(cfg.d_model, patch=cfg.freq_patch)
        n_res = cfg.resnet_grid * cfg.resnet_grid
        n_vit = (cfg.image_size // cfg.vit_patch) ** 2
        n_freq = (cfg.image_size // cfg.freq_patch) ** 2
        self.fusion = TokenFusion(cfg.d_model, n_res, n_vit, n_freq)
        self.num_tokens = n_res + n_vit + n_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_res = self.cnn(x)
        t_vit = self.vit(x)
        t_freq = self.freq(x)
        return self.fusion(t_res, t_vit, t_freq)  # (B, 441, d)
