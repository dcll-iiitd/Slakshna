"""
bhaskera.trainer.optimizer_registry
===================================
Registry for custom user-provided optimizer plugins.
"""
from __future__ import annotations
import logging
from typing import Callable, Dict

logger = logging.getLogger(__name__)

OPTIMIZER_REGISTRY: Dict[str, Callable] = {}

def register_optimizer(name: str) -> Callable:
    """Decorator to register a custom optimizer plugin."""
    def decorator(fn: Callable) -> Callable:
        if name in OPTIMIZER_REGISTRY:
            logger.warning(f"Overwriting existing optimizer plugin: '{name}'")
        OPTIMIZER_REGISTRY[name] = fn
        return fn
    return decorator
