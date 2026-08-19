"""
bhaskera.trainer.optim
======================
Optimizer and LR-scheduler factories.
Changes vs v1:
  * Pluggable Optimizer System (Registry, Torch Native, or Default)
  * Parameter grouping: weight_decay applied only to 2-D+ tensors
    (i.e. not to bias / LayerNorm / embeddings), following the standard
    GPT training recipe.
"""
from __future__ import annotations

import logging

import torch
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)

from .optimizer_registry import OPTIMIZER_REGISTRY

logger = logging.getLogger(__name__)


def _get_default_param_groups(model: torch.nn.Module, weight_decay: float):
    """Heuristic: 2-D+ params -> decay; 1-D (bias, LN gain, etc.) -> no decay."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: torch.nn.Module, train_cfg) -> torch.optim.Optimizer:
    """
    Builds the optimizer based on the specified backend (default, torch, plugin).
    """
    opt_cfg = getattr(train_cfg, "optimizer", None)
    
    # Backward compatibility safeguard
    if opt_cfg is None or opt_cfg.backend == "default":
        param_groups = _get_default_param_groups(model, train_cfg.weight_decay)
        fused = torch.cuda.is_available()
        
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info("Optimizer: AdamW (Default) | Source: bhaskera.trainer.optim")
            
        return AdamW(
            param_groups,
            lr=train_cfg.lr,
            betas=(0.9, 0.95),
            fused=fused,
        )

    elif opt_cfg.backend == "torch":
        cls_name = opt_cfg.class_name
        if not cls_name or not hasattr(torch.optim, cls_name):
            available = [n for n in dir(torch.optim) if isinstance(getattr(torch.optim, n), type) and issubclass(getattr(torch.optim, n), torch.optim.Optimizer)]
            raise ValueError(
                f"Unknown torch optimizer '{cls_name}'.\nAvailable optimizers:\n  " + 
                "\n  ".join(available)
            )
            
        opt_cls = getattr(torch.optim, cls_name)
        weight_decay = opt_cfg.kwargs.get("weight_decay", train_cfg.weight_decay)
        param_groups = _get_default_param_groups(model, weight_decay)
        
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(f"Optimizer: {cls_name} | Source: torch.optim")
            
        # Filter out weight_decay from kwargs to prevent double-passing
        kwargs = {k: v for k, v in opt_cfg.kwargs.items() if k != "weight_decay"}
        
        # Fallback to train_cfg.lr if not strictly defined in kwargs
        if "lr" not in kwargs:
            kwargs["lr"] = train_cfg.lr
            
        return opt_cls(param_groups, **kwargs)

    elif opt_cfg.backend == "plugin":
        name = opt_cfg.name
        if not name or name not in OPTIMIZER_REGISTRY:
            available = list(OPTIMIZER_REGISTRY.keys())
            raise ValueError(
                f"Unknown plugin optimizer '{name}'.\nAvailable optimizers:\n  " + 
                "\n  ".join(available)
            )
            
        builder_fn = OPTIMIZER_REGISTRY[name]
        # In case the plugin builder needs the full train_cfg context
        result = builder_fn(model, train_cfg)
        
        if dist.is_initialized() and dist.get_rank() == 0:
            module_source = builder_fn.__module__
            logger.info(f"Optimizer: {name} | Source: plugin | Module: {module_source}")

        # Support returning (param_groups, optimizer) if users implement advanced grouping
        if isinstance(result, tuple) and len(result) == 2:
            return result[1]
        return result

    else:
        raise ValueError(f"Unknown optimizer backend: {opt_cfg.backend}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg) -> LRScheduler:
    if train_cfg.warmup_steps <= 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, train_cfg.max_steps),
        )

    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=train_cfg.warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, train_cfg.max_steps - train_cfg.warmup_steps),
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[train_cfg.warmup_steps],
    )
