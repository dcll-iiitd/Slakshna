"""
GaLore Optimizer Plugin for Bhaskera (Distributed: DDP & FSDP Safe)
===================================================================
Gradient Low-Rank Projection (GaLore).

Features:
- DDP & FSDP Safe: Projects gradients *after* communication finishes.
- FSDP Shard Aware: Dynamically caps SVD rank for tiny parameter shards.
- Memory Optimized: Staggered SVD, bf16 states, early p.grad destruction.
- Mixed Precision Safe: Safely casts fp32 updates back to bf16/fp16 for projection.

FSDP REQUIREMENT: You must wrap your model with `use_orig_params=True`.
"""
from __future__ import annotations

import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer

from bhaskera.trainer.optimizer_registry import register_optimizer

logger = logging.getLogger(__name__)

_DEFAULT_GALORE_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "w1", "w2", "w3", "fc1", "fc2",
)

# ---------------------------------------------------------------------------
# Core GaLore algorithm
# ---------------------------------------------------------------------------

class GaLoreProjector:
    def __init__(self, rank, update_proj_gap=200, scale=0.25, proj_type="std", layer_index=0):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.layer_index = layer_index
        self.ortho_matrix = None

    @staticmethod
    def _orthogonal_basis(tensor: torch.Tensor, rank: int, side: str) -> torch.Tensor:
        orig_dtype = tensor.dtype
        orig_device = tensor.device
        
        matrix = tensor.float() if orig_dtype != torch.float32 else tensor
        U, _, Vh = torch.linalg.svd(matrix, full_matrices=False)

        # FSDP SAFEGUARD: A shard might be smaller than the requested rank.
        actual_rank = min(rank, matrix.shape[0], matrix.shape[1])

        if side == "right":
            basis = Vh[:actual_rank, :]
        elif side == "left":
            basis = U[:, :actual_rank]
        else:
            raise ValueError("side must be 'left' or 'right'")

        result = basis.to(device=orig_device, dtype=orig_dtype)
        
        del U, Vh, matrix
        torch.cuda.empty_cache() 
        
        return result

    def project(self, full_rank_grad: torch.Tensor, step: int) -> torch.Tensor:
        wide = full_rank_grad.shape[0] < full_rank_grad.shape[1]
        side = "right" if wide else "left"

        if self.ortho_matrix is None or (step + self.layer_index) % self.update_proj_gap == 0:
            self.ortho_matrix = self._orthogonal_basis(full_rank_grad, self.rank, side)

        if side == "right":
            return full_rank_grad @ self.ortho_matrix.t()
        return self.ortho_matrix.t() @ full_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        # FSDP/Mixed Precision Fix: The update from Adam math is float32. 
        # Cast it back to the original parameter dtype (e.g., bfloat16) for matmul.
        low_rank_grad = low_rank_grad.to(self.ortho_matrix.dtype)
        
        # Dynamic shape checking instead of strict rank checking handles 
        # FSDP shards where actual_rank < self.rank seamlessly.
        if self.ortho_matrix.shape[0] < self.ortho_matrix.shape[1]:  
            # Wide matrix -> was 'right' side
            full = low_rank_grad @ self.ortho_matrix
        else:  
            # Tall matrix -> was 'left' side
            full = self.ortho_matrix @ low_rank_grad
            
        return full * self.scale


class GaLoreAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, correct_bias=True):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)
        
        # State tracker to ensure staggering remains deterministic across FSDP shards
        self._galore_layer_counter = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            use_galore = "rank" in group

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("GaLore does not support sparse gradients")

                state = self.state[p]
                
                # 1. Initialize State & Projector safely inside the optimizer bounds
                if "step" not in state:
                    state["step"] = 0
                    if use_galore:
                        state["projector"] = GaLoreProjector(
                            rank=group["rank"], 
                            update_proj_gap=group.get("update_proj_gap", 200),
                            scale=group.get("scale", 0.25),
                            proj_type=group.get("proj_type", "std"),
                            layer_index=self._galore_layer_counter
                        )
                        self._galore_layer_counter += 1

                # 2. Project gradient (post-DDP/FSDP sync) and instantly free full-rank grad
                if use_galore:
                    grad = state["projector"].project(p.grad, state["step"])
                    # Freeing p.grad here keeps peak memory flat during the optimizer loop
                    p.grad = None 
                else:
                    grad = p.grad

                # 3. Setup BF16 moving averages using the size of the projected grad
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(grad, dtype=torch.bfloat16)
                    state["exp_avg_sq"] = torch.zeros_like(grad, dtype=torch.bfloat16)

                state["step"] += 1

                # 4. Math in Float32, Storage in BFloat16
                exp_avg = state["exp_avg"].float()
                exp_avg_sq = state["exp_avg_sq"].float()
                grad_f32 = grad.float()

                exp_avg.mul_(beta1).add_(grad_f32, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_f32, grad_f32, value=1.0 - beta2)
                
                state["exp_avg"].copy_(exp_avg)
                state["exp_avg_sq"].copy_(exp_avg_sq)

                bc1 = 1.0 - beta1 ** state["step"] if group["correct_bias"] else 1.0
                bc2 = 1.0 - beta2 ** state["step"] if group["correct_bias"] else 1.0
                
                denom = (exp_avg_sq / bc2).sqrt().add_(group["eps"])
                step_size = group["lr"] / bc1

                update = exp_avg / denom
                
                # 5. Project back to full rank ONLY for the update application
                if use_galore:
                    update = state["projector"].project_back(update)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                p.add_(update, alpha=-step_size)

        return loss


# ---------------------------------------------------------------------------
# Bhaskera plugin wiring 
# ---------------------------------------------------------------------------

def _split_galore_params(model: nn.Module, target_suffixes: set[str]):
    galore_params, regular_params = [], []
    galore_ids = set()

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        short_name = module_name.split(".")[-1]
        if target_suffixes and short_name not in target_suffixes:
            continue
        if module.weight.requires_grad:
            galore_params.append(module.weight)
            galore_ids.add(id(module.weight))

    for _, p in model.named_parameters():
        if p.requires_grad and id(p) not in galore_ids:
            regular_params.append(p)

    return galore_params, regular_params


@register_optimizer("galore")
def build_galore(model, train_cfg):
    opt_cfg = train_cfg.optimizer
    kwargs = dict(opt_cfg.kwargs) 

    lr = kwargs.pop("lr", train_cfg.lr)
    weight_decay = kwargs.pop("weight_decay", train_cfg.weight_decay)
    rank = kwargs.pop("rank", 128)
    update_proj_gap = kwargs.pop("update_proj_gap", 200)
    scale = kwargs.pop("scale", 0.25)
    proj_type = kwargs.pop("proj_type", "std")
    target_suffixes = set(kwargs.pop("target_modules", _DEFAULT_GALORE_TARGETS))
    kwargs.pop("use_8bit", None) 

    galore_params, regular_params = _split_galore_params(model, target_suffixes)

    if not galore_params:
        raise ValueError("GaLore found no matching Linear weights to project.")

    param_groups = [
        {
            "params": galore_params,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
            "weight_decay": weight_decay,
        },
        {
            "params": regular_params,
            "weight_decay": weight_decay,
        },
    ]

    logger.info(
        f"GaLore (Distributed): {len(galore_params)} projected tensor(s) "
        f"(rank={rank}, staggered SVD, bf16 states) "
        f"+ {len(regular_params)} regular tensor(s)"
    )

    return GaLoreAdamW(param_groups, lr=lr, betas=(0.9, 0.999), **kwargs)
