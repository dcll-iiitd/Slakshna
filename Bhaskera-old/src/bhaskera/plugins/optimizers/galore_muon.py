"""
GaLore + Muon Optimizer Plugin for Bhaskera
===========================================
Combines GaLore's memory-efficient low-rank projection with 
Muon's 8-bit momentum and Newton-Schulz orthogonalization.

For large weight matrices, gradients are projected to a low-rank space, 
optimized using 8-bit Muon, and projected back. 
For 1D parameters or already low-rank matrices (e.g. LoRA), 
it safely falls back to standard AdamW.
"""
from __future__ import annotations

import logging
import torch
from torch.optim import Optimizer, AdamW

from bhaskera.trainer.optimizer_registry import register_optimizer
from bhaskera.plugins.optimizers.galore import GaLoreProjector
from bhaskera.plugins.optimizers.muon import Muon

logger = logging.getLogger(__name__)

class GaLoreMuon(Optimizer):
    def __init__(self, params, rank=128, update_proj_gap=200, scale=0.25, proj_type="std", 
                 lr=0.02, momentum=0.95, weight_decay=0.01, nesterov=True, 
                 ns_chunk_size=128, optim_bits=8, **kwargs):
        
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            betas=(momentum, 0.0),
            weight_decay=weight_decay, 
            nesterov=nesterov,
            adjust_lr="rms_norm",
            ns_chunk_size=ns_chunk_size,
            rank=rank,
            update_proj_gap=update_proj_gap,
            scale=scale,
            proj_type=proj_type,
            optim_bits=optim_bits
        )
        super().__init__(params, defaults)
        
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        
        # We will dynamically instantiate Muon and AdamW when step() is called 
        # for the first time, because we need to see the projected gradients 
        # to pass them to Muon.
        self.muon_opt = None
        self.adam_opt = None
        self._galore_layer_counter = 0

        # Create parameter groups for AdamW (bypassed params)
        self.adam_param_groups = []
        for group in self.param_groups:
            adam_group_params = []
            for p in group["params"]:
                is_1d = p.ndim < 2
                is_small = p.ndim >= 2 and min(p.shape[0], p.shape[1]) <= group["rank"]
                if is_1d or is_small:
                    adam_group_params.append(p)
            
            if adam_group_params:
                adam_group = {k: v for k, v in group.items() if k != "params"}
                adam_group["params"] = adam_group_params
                # Map Muon momentum to AdamW betas
                adam_group["betas"] = (group["momentum"], 0.999)
                self.adam_param_groups.append(adam_group)

        if self.adam_param_groups:
            self.adam_opt = AdamW(self.adam_param_groups)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        muon_param_groups = []
        dummy_to_orig = {}

        for group in self.param_groups:
            muon_group_params = []
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                is_1d = p.ndim < 2
                is_small = p.ndim >= 2 and min(p.shape[0], p.shape[1]) <= group["rank"]
                
                if is_1d or is_small:
                    continue # Handled by AdamW
                
                state = self.state[p]

                # Initialize Projector
                if "projector" not in state:
                    state["step"] = 0
                    state["projector"] = GaLoreProjector(
                        rank=group["rank"], 
                        update_proj_gap=group.get("update_proj_gap", 200),
                        scale=group.get("scale", 0.25),
                        proj_type=group.get("proj_type", "std"),
                        layer_index=self._galore_layer_counter
                    )
                    self._galore_layer_counter += 1

                state["step"] += 1

                # Project gradient
                grad_proj = state["projector"].project(p.grad, state["step"])
                p.grad = None # Free memory
                
                # Create a dummy parameter for Muon's 8-bit tracking.
                # Must be 2D for Muon.
                if "dummy_p" not in state:
                    dummy_p = torch.zeros_like(grad_proj, device=grad_proj.device, dtype=grad_proj.dtype)
                    state["dummy_p"] = dummy_p
                
                dummy_p = state["dummy_p"]
                dummy_p.grad = grad_proj
                
                muon_group_params.append(dummy_p)
                dummy_to_orig[id(dummy_p)] = p

            if muon_group_params:
                muon_group = {k: v for k, v in group.items() if k != "params"}
                muon_group["params"] = muon_group_params
                muon_group["adjust_lr"] = None # Disable Muon's internal shape-based LR scaling
                muon_param_groups.append(muon_group)

        # Initialize Muon lazily on first step when dummies are created
        if self.muon_opt is None and muon_param_groups:
            self.muon_opt = Muon(muon_param_groups)
            if hasattr(self, "_saved_muon_state") and self._saved_muon_state is not None:
                self.muon_opt.load_state_dict(self._saved_muon_state)
                self._saved_muon_state = None
            # Muon relies on its own self.param_groups internally, so we just pass it the initialized groups
            self.muon_opt.param_groups = muon_param_groups
        elif self.muon_opt is not None:
            self.muon_opt.param_groups = muon_param_groups

        # Step AdamW
        if self.adam_opt is not None:
            self.adam_opt.step()

        # Step Muon
        if self.muon_opt is not None:
            # Temporarily disable weight decay in Muon to apply it manually on full-rank
            wds = []
            for group in self.muon_opt.param_groups:
                wds.append(group["weight_decay"])
                group["weight_decay"] = 0.0
                
            self.muon_opt.step()
            
            # Restore weight decay and apply full-rank updates
            import math
            for group, wd in zip(self.muon_opt.param_groups, wds):
                lr = group["lr"]
                for dummy_p in group["params"]:
                    orig_p = dummy_to_orig[id(dummy_p)]
                    projector = self.state[orig_p]["projector"]
                    
                    # dummy_p.data contains the low-rank update 
                    # (since dummy_p was initialized to 0 and Muon applied the update to it)
                    low_rank_update = dummy_p.data.clone()
                    dummy_p.data.zero_() # Reset for next step
                    
                    # Project back to full rank
                    full_rank_update = projector.project_back(low_rank_update)
                    
                    # Compute scaling correction to match GaLore AdamW update magnitude.
                    # GaLore AdamW produces updates with elements roughly proportional to 1/sqrt(max(m, n)).
                    # Our full_rank_update has elements roughly proportional to 1/sqrt(m * n).
                    # Multiplying by sqrt(min(m, n)) ensures the GaLoreMuon step size matches 
                    # what the user's learning rate (e.g. 3e-4) was tuned for in GaLore AdamW.
                    m, n = orig_p.shape
                    correction_scale = math.sqrt(min(m, n))
                    
                    # Apply weight decay and update manually to orig_p
                    orig_p.data.mul_(1.0 - lr * wd).add_(full_rank_update, alpha=correction_scale)

        return loss

    def state_dict(self):
        state_dict = super().state_dict()
        
        # Sanitize projector objects to dicts so they can be serialized
        for p_id, state in state_dict["state"].items():
            if "projector" in state:
                proj = state["projector"]
                state["projector"] = {
                    "__is_galore_projector": True,
                    "rank": proj.rank,
                    "update_proj_gap": proj.update_proj_gap,
                    "scale": proj.scale,
                    "proj_type": proj.proj_type,
                    "layer_index": proj.layer_index,
                    "ortho_matrix": proj.ortho_matrix,
                }
        
        # Add internal optimizer states
        state_dict["adam_state"] = self.adam_opt.state_dict() if self.adam_opt else None
        state_dict["muon_state"] = self.muon_opt.state_dict() if self.muon_opt else None
        return state_dict

    def load_state_dict(self, state_dict):
        self._saved_adam_state = state_dict.pop("adam_state", None)
        self._saved_muon_state = state_dict.pop("muon_state", None)
        
        # Restore projector objects before PyTorch loads them
        for p_id, state in state_dict["state"].items():
            if "projector" in state and isinstance(state["projector"], dict) and state["projector"].get("__is_galore_projector"):
                p_state = state["projector"]
                proj = GaLoreProjector(
                    rank=p_state["rank"],
                    update_proj_gap=p_state["update_proj_gap"],
                    scale=p_state["scale"],
                    proj_type=p_state["proj_type"],
                    layer_index=p_state["layer_index"]
                )
                proj.ortho_matrix = p_state["ortho_matrix"]
                state["projector"] = proj
                
        super().load_state_dict(state_dict)
        
        if self.adam_opt and self._saved_adam_state:
            self.adam_opt.load_state_dict(self._saved_adam_state)
            self._saved_adam_state = None

@register_optimizer("galore_muon")
def get_galore_muon(model, train_cfg):
    kwargs = getattr(train_cfg.optimizer, "kwargs", {})
    if "lr" not in kwargs:
        kwargs["lr"] = train_cfg.lr
    if "weight_decay" not in kwargs:
        kwargs["weight_decay"] = train_cfg.weight_decay
    return GaLoreMuon(model.parameters(), **kwargs)
