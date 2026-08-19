"""
bhaskera.launcher.diagnostics
==============================
Quick sanity check: NCCL connectivity + bfloat16 + bandwidth.

Usage (via Ray):
    python -m bhaskera.launcher.diagnostics --num-workers 4

Usage (via srun):
    srun --ntasks=4 python -m bhaskera.launcher.diagnostics --slurm
"""
from __future__ import annotations
import argparse
import os
import socket
import time

import torch
import torch.distributed as dist


def _diag_worker(_config):
    """Ray Train worker that runs diagnostics."""
    import ray.train
    ctx        = ray.train.get_context()
    rank       = ctx.get_world_rank()
    local_rank = ctx.get_local_rank()
    world_size = ctx.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # All-reduce test
    t = torch.ones(1, device=device, dtype=torch.bfloat16) * rank
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    assert abs(t.item() - expected) < 0.5, f"allreduce mismatch rank {rank}"

    # Bandwidth
    size = 25 * 1024 * 1024
    buf  = torch.ones(size, device=device)
    dist.barrier()
    t0 = time.time()
    for _ in range(5):
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    bw = 2 * (world_size - 1) / world_size * size * 4 * 5 / (time.time() - t0) / 1e9

    if rank == 0:
        props = torch.cuda.get_device_properties(device)
        print(f"\n{'='*50}")
        print(f"  Bhaskera diagnostics PASSED")
        print(f"  World size : {world_size}")
        print(f"  GPU        : {props.name} ({props.total_memory // 1024**3} GB)")
        print(f"  NCCL BW    : {bw:.2f} GB/s")
        print(f"{'='*50}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args()

    import ray
    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, RunConfig

    ray.init(ignore_reinit_error=True)
    num_workers = args.num_workers or torch.cuda.device_count()

    TorchTrainer(
        train_loop_per_worker=_diag_worker,
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=True),
        run_config=RunConfig(name="bhaskera-diagnostics"),
    ).fit()


if __name__ == "__main__":
    main()
