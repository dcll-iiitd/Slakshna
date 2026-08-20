"""
bhaskera.utils.gpu_stats
========================
Fast GPU telemetry via pynvml.

Why this replaces the old nvidia-smi subprocess call:
  * nvidia-smi invocation with XML parsing takes ~100-500 ms and has a
    10-second timeout on hang; called every log step it stalls training.
  * pynvml binds to libnvidia-ml directly — constant-time, no fork/exec.
  * Graceful degradation: if pynvml isn't installed or nvml init fails,
    we return {} silently so loggers don't crash.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_NVML_READY: Optional[bool] = None
_NVML = None


def _init_nvml() -> bool:
    global _NVML_READY, _NVML
    if _NVML_READY is not None:
        return _NVML_READY
    try:
        import pynvml
        pynvml.nvmlInit()
        _NVML = pynvml
        _NVML_READY = True
    except Exception as e:
        logger.debug(f"pynvml unavailable ({e}); GPU telemetry disabled")
        _NVML_READY = False
    return _NVML_READY


def gpu_stats() -> dict[str, float]:
    """Per-GPU utilisation, memory, temperature, power.  Returns {} on failure."""
    if not _init_nvml():
        return {}
    result: dict[str, float] = {}
    try:
        n = _NVML.nvmlDeviceGetCount()
        for i in range(n):
            h = _NVML.nvmlDeviceGetHandleByIndex(i)
            try:
                util = _NVML.nvmlDeviceGetUtilizationRates(h)
                result[f"gpu/{i}/util_pct"] = float(util.gpu)
            except Exception:
                pass
            try:
                mem = _NVML.nvmlDeviceGetMemoryInfo(h)
                result[f"gpu/{i}/mem_mib"] = float(mem.used) / (1024 * 1024)
            except Exception:
                pass
            try:
                temp = _NVML.nvmlDeviceGetTemperature(h, _NVML.NVML_TEMPERATURE_GPU)
                result[f"gpu/{i}/temp_c"] = float(temp)
            except Exception:
                pass
            try:
                power = _NVML.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW → W
                result[f"gpu/{i}/power_w"] = float(power)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"gpu_stats query failed: {e}")
    return result