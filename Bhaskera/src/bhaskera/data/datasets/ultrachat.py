
from __future__ import annotations

import ray.data

from bhaskera.data.registry import register, register_raw
from bhaskera.data.tokenize import tokenize_dataset


@register_raw("ultrachat", text_col="messages")
def _build_raw(cfg) -> ray.data.Dataset:
    """Return the raw (un-tokenized) UltraChat dataset."""
    from datasets import load_dataset
    hf_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    return ray.data.from_huggingface(hf_ds)


@register("ultrachat")
def build(cfg, world_size: int = 1) -> ray.data.Dataset:
    """Return the tokenized UltraChat dataset, using cache if available."""
    return tokenize_dataset(_build_raw(cfg), cfg, "messages", world_size=world_size)