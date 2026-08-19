"""
bhaskera.data.datasets.local_cpt
================================
Continual Pre-Training generic dataset builder for local files.
"""
from __future__ import annotations

import ray.data
from bhaskera.data.registry import register, register_raw
from bhaskera.data.tokenize import load_tokenized, tokenize_dataset
from .local_chat import _build_raw  # Reuse existing file discovery logic

@register_raw("local_cpt", text_col="text")
def _build_raw_cpt(cfg, split=None) -> ray.data.Dataset:
    return _build_raw(cfg, split)

@register("local_cpt")
def build(cfg, world_size: int = 1) -> ray.data.Dataset:
    if cfg.data.tokenized_path:
        return load_tokenized(cfg.data.tokenized_path, cfg, world_size)
    return tokenize_dataset(_build_raw_cpt(cfg), cfg, "text", world_size=world_size)
