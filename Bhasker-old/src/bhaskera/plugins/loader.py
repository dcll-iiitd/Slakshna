"""
bhaskera.plugins.loader
=======================
Dynamic loader for user plugins (optimizers, etc.).
"""
from __future__ import annotations
import importlib
import logging

logger = logging.getLogger(__name__)

def load_plugins(cfg) -> None:
    """Dynamically load plugins specified in the YAML configuration."""
    if not hasattr(cfg, "plugins") or not cfg.plugins:
        return
        
    for opt_plugin in getattr(cfg.plugins, "optimizers", []):
        try:
            importlib.import_module(opt_plugin)
            logger.debug(f"Successfully loaded optimizer plugin: {opt_plugin}")
        except Exception as e:
            logger.error(f"Failed to load plugin '{opt_plugin}': {e}")
            raise ImportError(f"Could not load optimizer plugin '{opt_plugin}'.") from e
