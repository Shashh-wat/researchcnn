"""Minimal DDP helpers. No-op gracefully on single-process / non-CUDA runs (CPU/MPS dev machines)."""
import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_setup() -> tuple[int, int, int, torch.device]:
    """Reads RANK/WORLD_SIZE/LOCAL_RANK set by `torchrun`. Falls back to single-process otherwise."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return rank, world_size, local_rank, device

    device = pick_device()
    return 0, 1, 0, device


def ddp_cleanup():
    if is_distributed():
        dist.destroy_process_group()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_main_process() -> bool:
    if not is_distributed():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def all_gather_concat(tensor: torch.Tensor) -> torch.Tensor:
    """Gathers a tensor from all ranks and concatenates along dim 0. No-op if not distributed."""
    if not is_distributed():
        return tensor
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)
