
from __future__ import annotations

import ray.data

from bhaskera.data.registry import register, register_raw
from bhaskera.data.tokenize import tokenize_dataset


@register_raw("openassistant", text_col="text")
def _build_raw(cfg) -> ray.data.Dataset:
    """Return the raw (un-tokenized) OpenAssistant dataset."""
    from datasets import load_dataset
    hf_ds = load_dataset("timdettmers/openassistant-guanaco", split="train")
    return ray.data.from_huggingface(hf_ds)


@register("openassistant")
def build(cfg, world_size: int = 1) -> ray.data.Dataset:
    """Return the tokenized OpenAssistant dataset, using cache if available."""
    return tokenize_dataset(_build_raw(cfg), cfg, "text", world_size=world_size)