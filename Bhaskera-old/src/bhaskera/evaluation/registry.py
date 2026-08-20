import importlib
import logging
from typing import Callable, Optional, Dict

logger = logging.getLogger(__name__)

METRIC_REGISTRY: Dict[str, type] = {}
BENCHMARK_REGISTRY: Dict[str, type] = {}

def register_metric(name: str) -> Callable:
    """Decorator to register a validation metric."""
    def wrapper(cls: type) -> type:
        if name in METRIC_REGISTRY:
            logger.warning(f"Overwriting metric plugin: '{name}'")
        METRIC_REGISTRY[name] = cls
        return cls
    return wrapper

def register_benchmark(name: str) -> Callable:
    """Decorator to register a benchmark."""
    def wrapper(cls: type) -> type:
        if name in BENCHMARK_REGISTRY:
            logger.warning(f"Overwriting benchmark plugin: '{name}'")
        BENCHMARK_REGISTRY[name] = cls
        return cls
    return wrapper

def get_metric(name: str) -> Optional[type]:
    """Lazy-load the metric plugin if not already registered."""
    if name not in METRIC_REGISTRY:
        try:
            importlib.import_module(f"bhaskera.plugins.metrics.{name}")
        except ImportError as e:
            logger.warning(f"Could not lazy-load metric plugin '{name}': {e}")
    return METRIC_REGISTRY.get(name)

def get_benchmark(name: str) -> Optional[type]:
    """Lazy-load the benchmark plugin if not already registered."""
    if name not in BENCHMARK_REGISTRY:
        try:
            importlib.import_module(f"bhaskera.plugins.benchmarks.{name}")
        except ImportError as e:
            logger.warning(f"Could not lazy-load benchmark plugin '{name}': {e}")
    return BENCHMARK_REGISTRY.get(name)
