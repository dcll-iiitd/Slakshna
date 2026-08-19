"""
bhaskera.data.registry
======================
Decorator-based registry for Ray-Data dataset builders.

Phase 1 changes:
  fix #6 — Added RAW_REGISTRY and TEXT_COL alongside the existing REGISTRY.
            RAW_REGISTRY holds builders that return un-tokenized datasets.
            TEXT_COL maps dataset name → the column to tokenize.
            The @register_raw decorator is used by the bhaskera-tokenize CLI.
  fix #10 — build_ray_dataset now accepts world_size so partitioning is
            world-size-aware.


 

Adding a new dataset:

    from bhaskera.data.registry import register, register_raw

    @register_raw("my_dataset", text_col="text")
    def _build_raw(cfg, split=None) -> ray.data.Dataset:   # split is optional
        ...
        return ray.data.Dataset   # raw (not tokenized)

    @register("my_dataset")
    def build(cfg, world_size: int = 1) -> ray.data.Dataset:
        ...
        return tokenize_dataset(_build_raw(cfg), cfg, "text", world_size=world_size)
"""
from __future__ import annotations

import inspect
import logging
from typing import Callable, Dict, Optional

import ray.data

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Tokenized builder registry — same as before
REGISTRY: Dict[str, Callable] = {}

# fix #6: raw (un-tokenized) builder registry — used by bhaskera-tokenize CLI
RAW_REGISTRY: Dict[str, Callable] = {}

# fix #6: maps dataset name → column name that contains the text to tokenize
TEXT_COL: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

def register(name: str):
    """Decorator to register a tokenized dataset builder under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        if name in REGISTRY:
            logger.warning(f"Overwriting tokenized dataset registration for '{name}'")
        REGISTRY[name] = fn
        return fn
    return _wrap


def register_raw(name: str, text_col: str):
    """
    Decorator to register a raw (un-tokenized) dataset builder.

    Also records the text column name so bhaskera-tokenize can call
    persist_tokenized() without needing extra arguments.

    Usage:
        @register_raw("my_dataset", text_col="prompt")
        def _build_raw(cfg) -> ray.data.Dataset:
            ...

    The builder may also take an optional ``split`` kwarg if it cares about
    train/val. The launcher passes split through ``call_raw_builder``.
    """
    def _wrap(fn: Callable) -> Callable:
        if name in RAW_REGISTRY:
            logger.warning(f"Overwriting raw dataset registration for '{name}'")
        RAW_REGISTRY[name] = fn
        TEXT_COL[name]     = text_col
        return fn
    return _wrap


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_raw_builder(name: str, cfg, split: Optional[str] = None) -> ray.data.Dataset:
    """
    Call a raw builder, forwarding ``split`` only if it accepts that kwarg.

    This keeps existing builders (which don't know about splits) working
    unchanged, while letting new local builders separate train/val cleanly.
    """
    if name not in RAW_REGISTRY:
        raise ValueError(
            f"Dataset '{name}' is not registered in RAW_REGISTRY. "
            f"Available: {sorted(RAW_REGISTRY)}."
        )
    fn = RAW_REGISTRY[name]
    sig = inspect.signature(fn)
    kwargs = {}
    if split is not None and "split" in sig.parameters:
        kwargs["split"] = split
    elif split is not None and split != "train":
        # Builder ignores split, but caller asked for something other than
        # train — warn so the user notices their val_path won't be used.
        logger.warning(
            f"Builder '{name}' does not accept a 'split' argument; "
            f"ignoring split={split!r}."
        )
    return fn(cfg, **kwargs)


def build_ray_dataset(cfg, world_size: int = 1) -> ray.data.Dataset:
    """
    Build a tokenized Ray Dataset for the dataset named in cfg.data.name.

    fix #10: world_size is now propagated to the builder so partitioning
             is world-size-aware (prevents empty or unequal shards).
    """
    name = cfg.data.name
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(REGISTRY)}. "
            "Register yours with @register('name')."
        )
    logger.info(f"Building Ray dataset: '{name}' (world_size={world_size})")
    return REGISTRY[name](cfg, world_size=world_size)
