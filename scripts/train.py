"""GPU training entry point. Single-GPU: `python scripts/train.py ...`
Multi-GPU (DDP): `torchrun --nproc_per_node=N scripts/train.py ...`

Data layout expected (architecture.md / dataset.py):
    --train-manifest real_train.csv   (columns: path,label)
    --train-synth-dir synth_train/
    --val-manifest   real_val.csv
    --val-synth-dir  synth_val/
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import Config
from src.dataset import RealSyntheticSetDataset
from src.distributed import (
    all_gather_concat,
    ddp_cleanup,
    ddp_setup,
    get_rank,
    get_world_size,
    is_main_process,
)
from src.losses import CombinedLoss
from src.metrics import compute_metrics
from src.model import MembershipFingerprintNet
from src.sampler import ClassBalancedBatchSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--train-synth-dir", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--val-synth-dir", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--k-draws", type=int, default=None)
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["bf16", "fp16"], default=None)
    p.add_argument("--torch-compile", action="store_true")
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--num-workers", type=int, default=None)
    return p.parse_args()


def build_config(args) -> Config:
    cfg = Config()
    overrides = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "k_draws": args.k_draws, "d_model": args.d_model, "amp_dtype": args.amp_dtype,
        "ckpt_dir": args.ckpt_dir, "resume_from": args.resume_from, "num_workers": args.num_workers,
    }
    for key, val in overrides.items():
        if val is not None:
            setattr(cfg, key, val)
    if args.no_amp:
        cfg.use_amp = False
    if args.torch_compile:
        cfg.use_torch_compile = True
    if args.grad_checkpointing:
        cfg.use_grad_checkpointing = True
    if args.no_pretrained:
        cfg.pretrained_backbones = False
    return cfg


def build_loader(manifest, synth_dir, cfg, world_size, rank, train: bool):
    ds = RealSyntheticSetDataset(manifest, synth_dir, k=cfg.k_draws, image_size=cfg.image_size)
    if train:
        labels = [label for _, label in ds.real_items]
        sampler = ClassBalancedBatchSampler(labels, cfg.batch_size, world_size, rank, seed=cfg.seed)
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=cfg.num_workers > 0, drop_last=True,
        )
    else:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0,
        )
    return ds, loader, (sampler if train else None)


def cosine_warmup(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())


def move_to_device(batch, device, channels_last: bool):
    real_img, synth_imgs, labels = batch
    real_img = real_img.to(device, non_blocking=True)
    synth_imgs = synth_imgs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        real_img = real_img.to(memory_format=torch.channels_last)
    return real_img, synth_imgs, labels


@torch.no_grad()
def evaluate(model, loader, device, cfg, autocast_dtype):
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        real_img, synth_imgs, labels = move_to_device(batch, device, cfg.channels_last)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=cfg.use_amp and device.type == "cuda"):
            out = model(real_img, synth_imgs, labels)
        all_scores.append(out["score"].float())
        all_labels.append(labels)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)
    scores = all_gather_concat(scores).cpu().numpy()
    labels = all_gather_concat(labels).cpu().numpy()
    model.train()
    return compute_metrics(scores, labels)


def main():
    args = parse_args()
    cfg = build_config(args)

    rank, world_size, local_rank, device = ddp_setup()
    torch.manual_seed(cfg.seed + rank)

    train_ds, train_loader, train_sampler = build_loader(
        args.train_manifest, args.train_synth_dir, cfg, world_size, rank, train=True
    )
    _, val_loader, _ = build_loader(
        args.val_manifest, args.val_synth_dir, cfg, world_size, rank, train=False
    )

    model = MembershipFingerprintNet(cfg).to(device)
    if cfg.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if cfg.use_torch_compile:
        model = torch.compile(model)
    if world_size > 1:
        find_unused = True  # arcface head only used when labels given; harmless with DDP
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    loss_fn = CombinedLoss(cfg.lambda_bce, cfg.lambda_arcface, cfg.lambda_supcon, cfg.supcon_temperature)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = LambdaLR(optimizer, lr_lambda=lambda s: cosine_warmup(s, cfg.warmup_steps, total_steps))

    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        device.type if device.type == "cuda" else "cpu",
        enabled=cfg.use_amp and cfg.amp_dtype == "fp16" and device.type == "cuda",
    )

    start_epoch, best_f1 = 0, -1.0
    if cfg.resume_from:
        ckpt = load_checkpoint(cfg.resume_from, model, optimizer, scheduler, scaler, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_metric", -1.0)

    global_step = 0
    for epoch in range(start_epoch, cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            real_img, synth_imgs, labels = move_to_device(batch, device, cfg.channels_last)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.use_amp and device.type == "cuda"):
                out = model(real_img, synth_imgs, labels)
                losses = loss_fn(out, labels)
                loss = losses["total"] / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += losses["total"].item()

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if is_main_process() and global_step % cfg.log_every == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"epoch {epoch} step {global_step}/{total_steps} "
                          f"loss {running_loss / (step + 1):.4f} lr {lr_now:.2e}", flush=True)

        if is_main_process():
            print(f"epoch {epoch} done in {time.time() - t0:.1f}s, "
                  f"mean train loss {running_loss / len(train_loader):.4f}", flush=True)

        if (epoch + 1) % cfg.val_every == 0:
            metrics = evaluate(model, val_loader, device, cfg, amp_dtype)
            if is_main_process():
                print(f"[val] epoch {epoch}: {metrics}", flush=True)
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    save_checkpoint(f"{cfg.ckpt_dir}/best.pt", model, optimizer, scheduler, scaler, epoch, best_f1, cfg)
                save_checkpoint(f"{cfg.ckpt_dir}/last.pt", model, optimizer, scheduler, scaler, epoch, best_f1, cfg)

    ddp_cleanup()


if __name__ == "__main__":
    main()
