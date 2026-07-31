"""
bhaskera.launcher.train
=======================
Unified CLI + Ray Train driver.
"""
from __future__ import annotations
import argparse
import logging
import os
import subprocess

import ray
import ray.data
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from bhaskera.config import load_config
from bhaskera.data import build_ray_dataset
from bhaskera.launcher.monitoring import setup_monitoring
from bhaskera.launcher.worker import worker_fn
from bhaskera.plugins.loader import load_plugins

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        import setproctitle
        setproctitle.setproctitle("RayTrainer")
    except ImportError:
        pass
    args = _parse_args()
    cfg  = load_config(args.config)
    
    load_plugins(cfg)

    if args.no_dashboard:
        cfg.monitoring.dashboard = False
    if args.dashboard_port:
        cfg.monitoring.dashboard_port = args.dashboard_port

    monitoring = setup_monitoring(cfg)

    _init_ray(monitoring)

    logger.info(monitoring.banner())

    num_workers = args.num_workers or _count_gpus()
    logger.info(f"Launching with {num_workers} GPU worker(s)")

    ray_dataset = build_ray_dataset(cfg, world_size=num_workers)

    datasets_dict = {"train": ray_dataset}
    if getattr(cfg, "evaluation", None) and cfg.evaluation.enabled:
        if getattr(cfg.data, "val_tokenized_path", None):
            try:
                from bhaskera.data.datasets.local_chat import build_val_ray_dataset
                val_ray_dataset = build_val_ray_dataset(cfg, world_size=num_workers)
                if val_ray_dataset:
                    datasets_dict["val"] = val_ray_dataset
            except ImportError:
                logger.warning("Could not import build_val_ray_dataset")

    # ── Dynamic Resource Allocation ──
    # Automatically allocate resources without hardcoding magic numbers.
    total_cpus = os.cpu_count() or 64
    data_workers = getattr(cfg.data, "num_workers", 0)
    
    # Ray Data operators (like Repartition/AllToAll) require extra unallocated CPUs to run.
    # We guarantee a safety cushion of either the YAML workers or 16 CPUs, whichever is larger.
    data_cushion = max(data_workers, 16)
    
    # Reserve the cushion, then divide the remainder among the GPU training workers
    available_cpus = max(1, total_cpus - data_cushion)
    cpus_per_worker = max(1, available_cpus // num_workers)

    logger.info(
        f"Dynamic Resource Allocation | System: {total_cpus} CPUs, {num_workers} GPUs. "
        f"Reserved data cushion of {data_cushion} CPUs (YAML workers: {data_workers}). "
        f"Allocating {cpus_per_worker} CPUs per GPU training worker."
    )

    trainer = TorchTrainer(
        train_loop_per_worker=worker_fn,
        train_loop_config=cfg.as_dict(),
        datasets=datasets_dict,
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={
                "GPU": 1,
                "CPU": cpus_per_worker,  # Dynamically bounded to guarantee headroom
            },
        ),
        run_config=RunConfig(
            name=cfg.logging.run_name,
            storage_path=os.path.abspath(args.storage_path or cfg.checkpoint.save_dir),
            checkpoint_config=CheckpointConfig(num_to_keep=cfg.checkpoint.keep_last_n),
            failure_config=ray.train.FailureConfig(max_failures=args.max_failures),
        ),
    )

    result = trainer.fit()
    logger.info(f"Training finished | best checkpoint: {result.best_checkpoints}")


def _init_ray(monitoring) -> None:
    if ray.is_initialized():
        return

    slurm_address = os.environ.get("RAY_ADDRESS")

    if slurm_address:
        logger.info(f"Connecting to Ray cluster at {slurm_address}")
        ray.init(address=slurm_address)
    else:
        logger.info("Stopping any stale Ray session...")
        subprocess.run(["ray", "stop", "--force"], capture_output=True)

        for var in ("RAY_ADDRESS", "RAY_HEAD_SERVICE_HOST", "RAY_HEAD_SERVICE_PORT"):
            os.environ.pop(var, None)

        n_gpus = _count_gpus()
        logger.info(f"Starting local Ray cluster ({n_gpus} GPU(s))")

        init_kwargs = {
            "num_cpus": os.cpu_count(),
            "num_gpus": n_gpus,
        }
        init_kwargs.update(monitoring.ray_init_kwargs())
        ray.init(**init_kwargs)

    logger.info(f"Ray resources: {ray.available_resources()}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bhaskera training launcher")
    p.add_argument("--config",         required=True,          help="Path to YAML config")
    p.add_argument("--num-workers",    type=int, default=None, help="Number of GPU workers")
    p.add_argument("--max-failures",   type=int, default=2,    help="Ray fault tolerance")
    p.add_argument("--storage-path",   type=str, default=None, help="Ray Train storage path")
    p.add_argument("--no-dashboard",   action="store_true",    help="Disable Ray Dashboard")
    p.add_argument("--dashboard-port", type=int, default=None, help="Ray Dashboard port")
    return p.parse_args()


def _count_gpus() -> int:
    import torch
    slurm_nodes = int(os.environ.get("SLURM_NNODES", 0))
    slurm_gpus  = int(os.environ.get("SLURM_GPUS_PER_NODE", 0))

    if slurm_nodes > 0 and slurm_gpus > 0:
        total = slurm_nodes * slurm_gpus
        logger.info(f"SLURM GPU count: {slurm_nodes} nodes × {slurm_gpus} GPUs/node = {total} total")
        return total

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No GPUs found.")
    return n


if __name__ == "__main__":
    main()
