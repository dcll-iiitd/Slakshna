"""
Decision-quality evidence about a peer: does applying their update help us?

For a peer j with delta D_j, the utility is the drop in loss on a probe set when
D_j is added to the adapter we started the round from (exactly what aggregation
would do with weight 1):

    loss_change_j = L(theta) - L(theta + D_j)        > 0  means the peer helps

Two ways to get it:
  first_order  one forward+backward gives g = grad L(theta); then
               loss_change_j ~= -g . D_j  (a dot product per peer). Cheap and
               accurate for small deltas, which LoRA updates are.
  exact        one forward per peer at theta + D_j. K times the cost.

The probe set is a handful of rows of the node's tokenized data (its own
``val_tokenized_path`` if configured). Unless a separate validation set is
configured it overlaps data this node trained on, so it is a probe, not a clean
held-out set; it is still the same set for every peer, which is what matters for
ranking peers against each other.

The math is pure torch and testable on a toy model; only ``evaluate_peers`` touches
Bhaskera and a real model.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Mapping, Optional, Sequence

import torch

BatchLoss = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]
METHODS = ("first_order", "exact")


def normalise_key(key: str) -> str:
    """PEFT keys differ by wrapper prefix and adapter name; compare without them."""
    key = re.sub(r"^(module\.|_orig_mod\.)+", "", key)
    key = key.replace("._orig_mod.", ".")
    return re.sub(r"(lora_[AB])\.[^.]+\.(weight)", r"\1.\2", key)


def build_name_map(param_names: Sequence[str], sd_keys: Sequence[str]) -> dict[str, str]:
    """model parameter name -> state-dict key, for the LoRA tensors both sides have."""
    by_norm = {normalise_key(k): k for k in sd_keys}
    out: dict[str, str] = {}
    for name in param_names:
        if "lora_" not in name:
            continue
        if name in sd_keys:
            out[name] = name
        elif normalise_key(name) in by_norm:
            out[name] = by_norm[normalise_key(name)]
    return out


def heldout_utilities(
    batch_losses: Sequence[BatchLoss],
    base_params: Mapping[str, torch.Tensor],
    deltas: Mapping[str, Mapping[str, torch.Tensor]],
    method: str = "first_order",
) -> tuple[float, dict[str, float]]:
    """Return (base_loss, {peer: loss_change}) on the probe batches.

    ``batch_losses[b](params)`` must return a scalar loss for params that
    replace ``base_params`` (same keys). ``deltas[peer]`` may cover a subset of
    keys; missing keys count as zero."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    if not batch_losses:
        raise ValueError("need at least one probe batch")
    n = len(batch_losses)

    if method == "first_order":
        params = {k: v.detach().clone().float().requires_grad_(True) for k, v in base_params.items()}
        base = 0.0
        for bl in batch_losses:
            loss = bl(params) / n
            loss.backward()
            base += float(loss.detach())
        out: dict[str, float] = {}
        for peer, d in deltas.items():
            dot = 0.0
            for k, p in params.items():
                if k in d and p.grad is not None:
                    dot += float((p.grad * d[k].to(p.device, p.dtype)).sum())
            out[peer] = -dot
        return base, out

    with torch.no_grad():
        base = sum(float(bl(base_params)) for bl in batch_losses) / n
        out = {}
        for peer, d in deltas.items():
            shifted = {
                k: (v + d[k].to(v.device, v.dtype)) if k in d else v for k, v in base_params.items()
            }
            out[peer] = base - sum(float(bl(shifted)) for bl in batch_losses) / n
    return base, out


def load_probe_batches(
    tokenized_dir: str, n_rows: int, seed: int, max_len: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """n_rows single-row (input_ids, labels) batches from tokenized parquet."""
    import glob
    import os

    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(tokenized_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet files under {tokenized_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for path in files:
        table = pq.read_table(path, columns=["input_ids", "labels"])
        total = table.num_rows
        if total == 0:
            continue
        for i in rng.sample(range(total), min(total, n_rows - len(batches))):
            ids = table.column("input_ids")[i].as_py()[:max_len]
            lab = table.column("labels")[i].as_py()[:max_len]
            if len(ids) < 2 or all(x == -100 for x in lab):
                continue
            batches.append((torch.tensor([ids], dtype=torch.long), torch.tensor([lab], dtype=torch.long)))
        if len(batches) >= n_rows:
            break
    if not batches:
        raise ValueError("probe set is empty (all rows masked or too short)")
    return batches


def evaluate_peers(
    config_path: str,
    old_sd: Mapping[str, torch.Tensor],
    peer_deltas: Mapping[str, Mapping[str, torch.Tensor]],
    device,
    method: str = "first_order",
    n_batches: int = 4,
    seed: int = 0,
    max_len: int = 1024,
) -> tuple[float, dict[str, float]]:
    """Real-model entry point used by ml_engine.py. Raises on any problem so the
    caller can fall back to cosine-only trust; never leaves the model in memory."""
    from bhaskera.config import load_config
    from bhaskera.models.loader import build_model
    from torch.func import functional_call

    cfg = load_config(config_path)
    data_dir = getattr(cfg.data, "val_tokenized_path", None) or cfg.data.tokenized_path
    if not data_dir:
        raise ValueError("config has no tokenized data path")
    batches = load_probe_batches(data_dir, n_batches, seed, max_len)

    model, _ = build_model(cfg, torch.device(device))
    try:
        model.eval()
        pnames = [n for n, _ in model.named_parameters()]
        name_map = build_name_map(pnames, list(old_sd.keys()))
        if not name_map:
            raise RuntimeError("no LoRA parameters of the model matched the adapter state dict keys")
        dtypes = {n: p.dtype for n, p in model.named_parameters() if n in name_map}
        dev = next(model.parameters()).device

        base_params = {n: old_sd[k].detach().to(dev).float() for n, k in name_map.items()}
        deltas = {
            peer: {n: d[k].detach().float() for n, k in name_map.items() if k in d and torch.is_tensor(d[k])}
            for peer, d in peer_deltas.items()
        }

        def make(ids: torch.Tensor, labels: torch.Tensor) -> BatchLoss:
            ids, labels = ids.to(dev), labels.to(dev)

            def bl(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
                cast = {n: v.to(dtypes[n]) for n, v in params.items()}
                out = functional_call(model, cast, args=(), kwargs={"input_ids": ids, "labels": labels})
                return out.loss.float()

            return bl

        return heldout_utilities([make(i, l) for i, l in batches], base_params, deltas, method)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
