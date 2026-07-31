"""
bhaskera.models
===============
Model loading + introspection + optional LoRA.

Public API:
    build_model(cfg, device) -> (model, profile)
    register_model(name)     -> decorator for custom non-HF models
"""
from __future__ import annotations

from .loader import build_model, register_model

__all__ = ["build_model", "register_model"]