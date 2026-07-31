"""
bhaskera.inference.sampling
============================
Sampling utilities for autoregressive language model decoding.

Supports:
  - Greedy decoding (argmax)
  - Temperature scaling
  - Top-k filtering
  - Top-p (nucleus) filtering
  - Combined top-k + top-p + temperature sampling
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by temperature. temperature=1.0 is identity.

    Args:
        logits:      (..., vocab_size) float tensor.
        temperature: > 0; lower = more peaked, higher = more uniform.

    Returns:
        Scaled logits of the same shape.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if abs(temperature - 1.0) < 1e-6:
        return logits
    return logits / temperature


def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k largest.

    Sets non-top-k positions to -inf so they get probability ≈ 0 after softmax.

    Args:
        logits: (..., vocab_size) float tensor.
        top_k:  Number of top logits to keep. ≤ 0 means no filtering.

    Returns:
        Filtered logits, same shape.
    """
    if top_k <= 0:
        return logits
    vocab_size = logits.size(-1)
    k = min(top_k, vocab_size)
    # kth largest value (threshold)
    threshold = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep the smallest set of tokens whose
    cumulative probability ≥ top_p.

    Args:
        logits: (..., vocab_size) float tensor.
        top_p:  Cumulative probability threshold ∈ (0, 1]. 1.0 = no filtering.

    Returns:
        Filtered logits, same shape.
    """
    if top_p >= 1.0:
        return logits

    # Sort descending and compute cumulative softmax probabilities
    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # Remove tokens where cumulative prob exceeds top_p (shift by 1 to keep
    # the token that crosses the threshold)
    sorted_mask = cumulative_probs - sorted_probs > top_p
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to original ordering
    filtered = logits.clone()
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


# ---------------------------------------------------------------------------
# Sampling entry points
# ---------------------------------------------------------------------------

def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Return the argmax token id at each position.

    Args:
        logits: (batch, vocab_size) or (vocab_size,) float tensor.

    Returns:
        (batch,) or scalar int64 tensor.
    """
    return logits.argmax(dim=-1)


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    do_sample: bool = True,
) -> torch.Tensor:
    """Sample next token ids from logits with optional filtering.

    Args:
        logits:      (batch, vocab_size) float tensor (last-position logits).
        temperature: Scaling factor. 1.0 = no change.
        top_k:       Keep top-k logits. 0 = disabled.
        top_p:       Nucleus threshold. 1.0 = disabled.
        do_sample:   If False, returns greedy argmax ignoring temperature/k/p.

    Returns:
        (batch,) int64 tensor of sampled token ids.
    """
    if not do_sample:
        return greedy_sample(logits)

    # 1. Temperature
    logits = temperature_scale(logits, temperature)

    # 2. Top-k
    if top_k > 0:
        logits = top_k_filter(logits, top_k)

    # 3. Top-p
    if top_p < 1.0:
        logits = top_p_filter(logits, top_p)

    # 4. Sample
    probs = F.softmax(logits, dim=-1)
    # Clamp for numerical safety (very rarely needed after filters)
    probs = probs.clamp(min=0.0)
    psum = probs.sum(dim=-1, keepdim=True)
    probs = probs / psum.clamp(min=1e-9)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)
