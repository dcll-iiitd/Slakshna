"""
bhaskera.trainer.moe
====================
MoE auxiliary loss extraction, load balancing, and expert utilisation
metrics.

Design — fully architecture-agnostic:
  All routing decisions are driven by the ModelProfile populated by
  introspect.py (num_experts, num_shared_experts, experts_per_token).
  No model name or class name is ever checked here.

  The one non-trivial inference is router logit shape.  Different
  architectures emit different tensors in out.router_logits:

    Shape (T, E_total)  — full-vocabulary logits (Mixtral, Qwen2MoE, …)
                          T = tokens, E_total = num_experts (routed only,
                          shared experts have their own paths and are not
                          included in router logits)

    Shape (T, k)        — top-k selected scores only (e.g. Param2)
                          k = experts_per_token

    Shape (T, 1)        — scalar gate (rare; binary MoE)

  Additionally some models wrap each per-layer entry as a tuple
  (logits, indices) instead of a bare tensor.

  _normalize_router_logits() unwraps tuples.
  _infer_logit_kind()        classifies the shape using profile fields.
  _load_balancing_loss_from_logits() picks the right loss formula.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

from bhaskera.introspect import ModelProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal constants — readable labels for logit shape kinds
# ---------------------------------------------------------------------------

_KIND_FULL  = "full"   # (T, num_experts)  — softmax over all experts
_KIND_TOPK  = "topk"   # (T, k)            — scores for selected experts only
_KIND_GATE  = "gate"   # (T, 1)            — scalar gate weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_router_logits(router_logits) -> list[torch.Tensor]:
    """
    Return a flat list of Tensors from whatever router_logits contains.

    Handles:
      - tuple/list of Tensors           (standard HF: one tensor per layer)
      - tuple/list of (Tensor, Tensor)  (some models return (logits, indices))
      - a single bare Tensor            (single-layer or already-stacked)
    """
    if isinstance(router_logits, torch.Tensor):
        # Already a single tensor — wrap so callers can always iterate
        return [router_logits]

    out: list[torch.Tensor] = []
    for item in router_logits:
        if isinstance(item, torch.Tensor):
            out.append(item)
        elif isinstance(item, (tuple, list)) and len(item) > 0:
            # (logits_tensor, indices_tensor) — take logits (first element)
            if isinstance(item[0], torch.Tensor):
                out.append(item[0])
            # else: unrecognised structure, skip silently
    return out


def _infer_logit_kind(logits: torch.Tensor, profile: ModelProfile) -> str:
    """
    Classify a single per-layer router logit tensor by its last dimension.

    Uses only profile fields (num_experts, num_shared_experts,
    experts_per_token) — no model-name checks.

    Priority:
      1. If last dim == 1                         → _KIND_GATE
      2. If last dim == num_experts (routed only) → _KIND_FULL
      3. If last dim == num_experts + num_shared  → _KIND_FULL
         (some architectures include shared in the gate)
      4. If last dim == experts_per_token         → _KIND_TOPK
      5. Fallback heuristic: small last dim
         (≤ max(experts_per_token * 2, 8))        → _KIND_TOPK
      6. Otherwise                                → _KIND_FULL
         (assume full-vocab; loss func will handle it)
    """
    n = logits.shape[-1]

    if n == 1:
        return _KIND_GATE

    total_routed = profile.num_experts                              # e.g. 64
    total_with_shared = total_routed + profile.num_shared_experts   # e.g. 66
    k = profile.experts_per_token                                   # e.g. 6

    if total_routed > 0 and n == total_routed:
        return _KIND_FULL
    if total_with_shared > 0 and n == total_with_shared:
        return _KIND_FULL
    if k > 0 and n == k:
        return _KIND_TOPK

    # Heuristic for architectures whose profile.experts_per_token was not
    # reliably detected (defaults to 2 in introspect.py).  A small last dim
    # that is clearly not the full expert count → treat as top-k.
    topk_threshold = max(k * 2, 8)
    if total_routed > 0 and n < total_routed and n <= topk_threshold:
        return _KIND_TOPK

    return _KIND_FULL  # safe default for unknown architectures


# ---------------------------------------------------------------------------
# Aux loss extraction (public entry point)
# ---------------------------------------------------------------------------

def extract_aux_loss(out, profile: ModelProfile) -> Optional[torch.Tensor]:
    """
    Pull the auxiliary loss out of the model output.

    Tries well-known attribute names first (so models that compute the loss
    themselves are zero-overhead), then falls back to computing it from
    router logits.  Returns None for dense models.
    """
    if not profile.is_moe:
        return None

    for attr in ("aux_loss", "router_aux_loss", "moe_loss", "load_balancing_loss"):
        val = getattr(out, attr, None)
        if isinstance(val, torch.Tensor):
            return val

    router_logits = getattr(out, "router_logits", None)
    if router_logits is not None:
        return _load_balancing_loss_from_logits(router_logits, profile)

    return None


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def _load_balancing_loss_from_logits(
    router_logits,
    profile: ModelProfile,
) -> Optional[torch.Tensor]:
    """
    Switch-Transformer-style load-balancing loss, averaged across layers.

    For each layer:
        loss_layer = N * sum_i( f_i * P_i )
    where
        f_i = fraction of tokens dispatched to expert i  (no grad)
        P_i = mean routing probability for expert i      (grad)
        N   = number of experts considered in this layer

    Works for all three logit shapes (_KIND_FULL / _KIND_TOPK / _KIND_GATE)
    without any model-name checks — shape classification is driven entirely
    by the ModelProfile.
    """
    normalized = _normalize_router_logits(router_logits)
    if not normalized:
        return None

    try:
        per_layer: list[torch.Tensor] = []

        for logits in normalized:
            kind = _infer_logit_kind(logits, profile)

            if kind == _KIND_GATE:
                # Scalar gate: single expert per token, nothing to balance
                # across experts — skip contributing to the loss.
                continue

            elif kind == _KIND_FULL:
                # Standard case: softmax over all expert logits, then compute
                # token fractions via a hard top-k mask.
                probs = torch.softmax(logits, dim=-1)           # (T, E)
                n_experts = logits.shape[-1]
                k = min(profile.experts_per_token or 1, n_experts)
                top_idx = probs.topk(k, dim=-1).indices         # (T, k)
                mask = torch.zeros_like(probs)
                mask.scatter_(1, top_idx, 1.0)                  # (T, E)
                f = mask.mean(dim=0)                            # (E,) no grad
                p = probs.mean(dim=0)                           # (E,) grad
                per_layer.append((f * p).sum() * n_experts)

            elif kind == _KIND_TOPK:
                # Router only emits scores for the already-selected k experts.
                # Every column corresponds to one selected expert, so the
                # "dispatch fraction" is uniform (1/k) across all columns.
                # We still let the probability term carry the gradient signal.
                probs = torch.softmax(logits, dim=-1)           # (T, k)
                n_cols = logits.shape[-1]                       # == k
                # f_i = 1/k for all i (uniform by construction)
                f = torch.full(
                    (n_cols,), 1.0 / n_cols,
                    dtype=probs.dtype, device=probs.device,
                )
                p = probs.mean(dim=0)                           # (k,) grad
                per_layer.append((f * p).sum() * n_cols)

        if not per_layer:
            return None
        return torch.stack(per_layer).mean()

    except Exception as e:
        logger.warning(f"Failed to compute load-balancing loss: {e}")
        return None


# ---------------------------------------------------------------------------
# Expert utilisation metrics (for W&B / MLflow logging)
# ---------------------------------------------------------------------------

def compute_expert_utilization(out, profile: ModelProfile) -> dict:
    """
    Compute per-expert load metrics from router logits.

    Returns a dict of scalars ready to pass to a logger:
        expert/load_max, expert/load_min, expert/load_std,
        expert/imbalance_ratio  (max/min, omitted when min == 0)

    Architecture-agnostic: uses _normalize_router_logits and
    _infer_logit_kind so it works for any model in ModelProfile.
    """
    metrics: dict = {}
    router_logits = getattr(out, "router_logits", None)
    if router_logits is None:
        return metrics

    try:
        normalized = _normalize_router_logits(router_logits)
        all_loads: list[torch.Tensor] = []

        for logits in normalized:
            kind = _infer_logit_kind(logits, profile)
            if kind == _KIND_GATE:
                continue

            probs = torch.softmax(logits.float(), dim=-1)       # (T, E or k)
            n_cols = logits.shape[-1]
            k = min(profile.experts_per_token or 1, n_cols)

            if kind == _KIND_FULL:
                top_idx = probs.topk(k, dim=-1).indices
                mask = torch.zeros_like(probs)
                mask.scatter_(1, top_idx, 1.0)
                all_loads.append(mask.sum(dim=0))               # (E,)
            else:  # _KIND_TOPK
                # Every token activates all k columns equally
                all_loads.append(
                    torch.ones(n_cols, dtype=probs.dtype, device=probs.device)
                    * probs.shape[0]                            # count = T
                )

        if not all_loads:
            return metrics

        avg_load = torch.stack(all_loads).mean(dim=0)
        total = avg_load.sum()
        if total <= 0:
            return metrics

        util = avg_load / total
        metrics["expert/load_max"] = util.max().item()
        metrics["expert/load_min"] = util.min().item()
        metrics["expert/load_std"] = util.std().item()
        min_val = util.min().item()
        if min_val > 0:
            metrics["expert/imbalance_ratio"] = util.max().item() / min_val

    except Exception as e:
        logger.debug(f"Expert utilisation computation failed: {e}")

    return metrics
