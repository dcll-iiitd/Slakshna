"""
Peer trust bookkeeping for ml_engine.py (pure Python, no torch).

Each node keeps a raw score ``alpha`` per peer; the aggregation weights are a
softmax of the scores. This module holds the pieces that decide how scores start,
how weights are restricted to the deltas that actually arrived, and how evidence
about a peer is turned into a score update.

Evidence per peer, both in [-1, 1]:
  cosine   how aligned the peer's update direction is with ours (cheap, always
           available, but says nothing about whether the update is any good);
  utility  how much applying the peer's update lowers the loss on a probe set
           (decision quality, see peer_eval.py in this package). May be unavailable.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Iterable, Mapping, Optional

INIT_MODES = ("uniform", "seeded", "random")


def init_alpha(mode: str, my_id: str, peer_id: str) -> float:
    """Starting score for a peer we have never exchanged anything with.

    uniform  0.0 for everyone: all peers start with equal weight (default).
    seeded   small deterministic perturbation from (my_id, peer_id): reproducible
             runs, still symmetric in expectation.
    random   the legacy behaviour, Normal(0, 1) from the global RNG. A single
             unlucky draw can hand a bad peer several times the weight of a good
             one, and the score only moves ~0.02 per round, so this noise
             dominates short runs.
    """
    if mode == "uniform":
        return 0.0
    if mode == "seeded":
        digest = hashlib.sha256(f"slakshna-trust-init|{my_id}|{peer_id}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big")).gauss(0.0, 0.1)
    if mode == "random":
        return random.gauss(0.0, 1.0)
    raise ValueError(f"unknown trust_init {mode!r}; expected one of {INIT_MODES}")


def softmax_weights(alpha: Mapping[str, float], nodes: Iterable[str]) -> dict[str, float]:
    nodes = list(nodes)
    vals = [alpha[n] for n in nodes]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    total = sum(exps)
    return {n: e / total for n, e in zip(nodes, exps)}


def renormalise(weights: Mapping[str, float], available: Iterable[str]) -> dict[str, float]:
    """Restrict ``weights`` to the nodes whose delta actually arrived and rescale
    to sum to 1. Without this, a missing peer's share simply vanishes and the
    aggregated update shrinks by that share. Falls back to uniform if every
    available weight is zero."""
    ids = [n for n in available if n in weights]
    if not ids:
        return {}
    total = sum(max(weights[n], 0.0) for n in ids)
    if total <= 0 or not math.isfinite(total):
        return {n: 1.0 / len(ids) for n in ids}
    return {n: max(weights[n], 0.0) / total for n in ids}


def utility_from_loss_change(loss_change: float, base_loss: float, rel_scale: float = 0.01) -> float:
    """Map "loss went down by loss_change" into [-1, 1].

    ``rel_scale`` is the relative improvement that counts as roughly full marks:
    with 0.01, lowering the probe loss by 1% of its value gives tanh(1) = 0.76.
    Negative means the peer's update made the probe loss worse."""
    if not (math.isfinite(loss_change) and math.isfinite(base_loss)) or base_loss <= 0 or rel_scale <= 0:
        return 0.0
    return math.tanh(loss_change / (rel_scale * base_loss))


def combine_signal(cosine: float, utility: Optional[float], decision_weight: float = 0.5) -> float:
    """Blend alignment and decision quality. Without a utility (no probe data,
    evaluation off or failed) the cosine alone is used, as before."""
    cosine = max(-1.0, min(1.0, cosine))
    if utility is None:
        return cosine
    lam = max(0.0, min(1.0, decision_weight))
    return (1.0 - lam) * cosine + lam * max(-1.0, min(1.0, utility))


def update_alpha(
    alpha: dict[str, float],
    weights: Mapping[str, float],
    signals: Mapping[str, float],
    beta: float,
    decay: float = 0.98,
) -> None:
    """In place. Peers with evidence this round get ``beta * signal * w * (1 - w)``;
    every score then decays towards 0 (towards uniform trust), so a peer we
    stop hearing from is slowly forgiven rather than frozen."""
    for j in list(alpha):
        if j in signals and j in weights:
            alpha[j] += beta * signals[j] * weights[j] * (1.0 - weights[j])
        alpha[j] *= decay
