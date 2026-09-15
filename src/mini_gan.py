"""Minimal DCGAN, sized for fast CPU/MPS training on small images (e.g. 64x64).

This is intentionally small and generic — it exists only to produce a real synthetic
pool with known real_used provenance for the detector to be tested against, not to be
a good generative model. Architecture: Radford et al. 2015 (DCGAN), standard conv/
conv-transpose stack.
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64, image_size: int = 64, channels: int = 3, base: int = 32):
        super().__init__()
        assert image_size in (32, 64), "sized for 32 or 64 (4 upsampling stages)"
        n_up = 3 if image_size == 32 else 4
        layers = [
            nn.ConvTranspose2d(latent_dim, base * (2 ** (n_up - 1)), 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * (2 ** (n_up - 1))),
            nn.ReLU(inplace=True),
        ]
        ch = base * (2 ** (n_up - 1))
        for _ in range(n_up - 1):
            next_ch = ch // 2
            layers += [
                nn.ConvTranspose2d(ch, next_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(inplace=True),
            ]
            ch = next_ch
        layers += [nn.ConvTranspose2d(ch, channels, 4, 2, 1, bias=False), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.shape[0], self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, image_size: int = 64, channels: int = 3, base: int = 32):
        super().__init__()
        n_down = 3 if image_size == 32 else 4
        layers = [
            nn.Conv2d(channels, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base
        for _ in range(n_down - 1):
            next_ch = ch * 2
            layers += [
                nn.Conv2d(ch, next_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch
        layers += [nn.Conv2d(ch, 1, 4, 1, 0, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)
