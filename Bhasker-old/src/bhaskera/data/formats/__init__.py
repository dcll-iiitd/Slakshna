"""
bhaskera.data.formats
=====================
Pluggable "format renderers" that convert one raw row into a single rendered
string (ready to tokenize). One format = one schema convention.

A renderer signature is:

    fn(row: dict, tokenizer, options: dict) -> str

where:
  * row        — a single sample as a Python dict (keys are dataset columns)
  * tokenizer  — the HuggingFace tokenizer (used for apply_chat_template)
  * options    — free-form dict from cfg.data.format_options

Built-in formats (see ``builtins.py``):
  * "chatml"    — rows have ``messages: [{role, content}, ...]``
  * "alpaca"    — rows have ``instruction``, optional ``input``, ``output``
  * "sharegpt"  — rows have ``conversations: [{from, value}, ...]``

Adding your own (no repo changes — drop a file anywhere your code imports):

    from bhaskera.data.formats import register_format

    @register_format("my_custom")
    def render(row, tokenizer, options):
        # build whatever string layout you want
        return f"### Q: {row['question']}\\n### A: {row['answer']}"

Then set ``data.format: my_custom`` in your YAML.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# name -> renderer fn
FORMAT_REGISTRY: Dict[str, Callable[[dict, Any, dict], str]] = {}

# Module-level guard so we only import builtins once even if called repeatedly.
_BUILTINS_LOADED = False


def register_format(name: str) -> Callable:
    """Decorator: register a format renderer under ``name``."""
    def _wrap(fn: Callable) -> Callable:
        if name in FORMAT_REGISTRY:
            logger.warning(f"Overwriting format renderer for '{name}'")
        FORMAT_REGISTRY[name] = fn
        return fn
    return _wrap


def _ensure_builtins_loaded() -> None:
    """
    Import the built-in renderers on demand.

    We import lazily because Ray pickles ``_TokenizerActorFactory`` and ships
    it to workers; importing inside the factory's ``__init__`` keeps the
    pickled object small and avoids HF imports at registry-import time.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import builtins  # noqa: F401  (side-effect: @register_format calls)
    _BUILTINS_LOADED = True


def render_with_format(
    name: str,
    row: dict,
    tokenizer: Any,
    options: Optional[dict] = None,
) -> str:
    """Look up ``name`` in FORMAT_REGISTRY and call it on one row."""
    _ensure_builtins_loaded()
    if name not in FORMAT_REGISTRY:
        raise ValueError(
            f"Unknown format '{name}'. "
            f"Available: {sorted(FORMAT_REGISTRY)}. "
            "Register yours with @register_format('name')."
        )
    return FORMAT_REGISTRY[name](row, tokenizer, options or {})


__all__ = [
    "FORMAT_REGISTRY",
    "register_format",
    "render_with_format",
]
