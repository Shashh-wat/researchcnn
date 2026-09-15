"""Train the tiny DCGAN (src/mini_gan.py) on a folder of real images, then sample a
synthetic pool from it. This is the "GAN" step the detector pipeline needs external
ground truth from — see novelty.md/architecture.md for why the detector itself contains
no GAN.

Run: python scripts/train_mini_gan.py --real-dir mini_demo/data/real_used \
        --out-dir mini_demo/data --image-size 64 --iterations 600 --n-samples 300
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mini_gan import Discriminator, Generator


class FlatImageFolder(Dataset):
    def __init__(self, image_dir: str, image_size: int):
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in exts)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # match generator's Tanh output range
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-dir", required=True, help="Folder of real_used images (GAN trains on this only)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--image-size", type=int, default=64, choices=[32, 64])
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--iterations", type=int, default=600)
    p.add_argument("--n-samples", type=int, default=300, help="Synthetic images to generate after training")
    p.add_argument("--lr", type=float, default=2e-4)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ds = FlatImageFolder(args.real_dir, args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"training on {len(ds)} real images from {args.real_dir}, device={device}")

    G = Generator(args.latent_dim, args.image_size).to(device)
    D = Discriminator(args.image_size).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    step = 0
    t0 = time.time()
    while step < args.iterations:
        for real in loader:
            if step >= args.iterations:
                break
            real = real.to(device)
            b = real.shape[0]
            z = torch.randn(b, args.latent_dim, device=device)

            # discriminator step
            opt_d.zero_grad(set_to_none=True)
            fake = G(z).detach()
            d_real = D(real)
            d_fake = D(fake)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            opt_d.step()

            # generator step
            opt_g.zero_grad(set_to_none=True)
            fake = G(z)
            d_fake_for_g = D(fake)
            g_loss = bce(d_fake_for_g, torch.ones_like(d_fake_for_g))
            g_loss.backward()
            opt_g.step()

            if step % 50 == 0:
                print(f"step {step}/{args.iterations} d_loss {d_loss.item():.3f} g_loss {g_loss.item():.3f} "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)
            step += 1

    print(f"GAN training done in {time.time() - t0:.1f}s")

    synth_dir = Path(args.out_dir) / "synth_pool"
    synth_dir.mkdir(parents=True, exist_ok=True)
    G.eval()
    with torch.no_grad():
        made = 0
        while made < args.n_samples:
            b = min(64, args.n_samples - made)
            z = torch.randn(b, args.latent_dim, device=device)
            imgs = (G(z).clamp(-1, 1) + 1) / 2  # back to [0,1]
            for i in range(b):
                arr = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                Image.fromarray(arr).save(synth_dir / f"synth_{made + i:05d}.png")
            made += b
    print(f"wrote {args.n_samples} synthetic images to {synth_dir}")


if __name__ == "__main__":
    main()
