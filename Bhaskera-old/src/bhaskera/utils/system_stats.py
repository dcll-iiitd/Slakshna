"""
bhaskera.utils.system_stats
===========================
Production-grade telemetry for LLM fine-tuning.

This is the superset that supersedes ``gpu_stats.py``.  ``gpu_stats.py``
is preserved for backward compatibility; new code should call
``system_stats()``.

Coverage:
    GPU (per-device, via pynvml):
        utilisation, memory used / free / total, temperature, power,
        fan, clock (SM / mem), PCIe Tx/Rx KB/s, NVLink Tx/Rx KB/s,
        ECC volatile/aggregate single- and double-bit error counts,
        throttle reasons, performance state.

    CPU / host (via psutil):
        per-process CPU%, system CPU% (mean and per-core),
        load average, memory used / available, swap used,
        disk I/O bytes/s, network I/O bytes/s, open file descriptors.

Design rules:
    * **Cheap.**  Initialised once.  Each call is O(num_gpus + 4).
      No fork/exec.  Safe to call every step at scale.
    * **Graceful degradation.**  If pynvml or psutil are missing,
      the corresponding sub-tree is silently dropped.  Returns {}
      rather than crashing the trainer.
    * **Stable keys.**  Metric names are flat, slash-separated, and
      compatible with what ``RayMetricsLogger`` will translate to
      Prometheus.  Per-GPU keys carry ``gpu/<idx>/...`` so tag
      extraction is unambiguous.
    * **Rate-derived counters are buffered.**  Disk and network I/O
      are exposed as B/s, computed from a delta against the previous
      call — the first call returns 0 to avoid a spurious initial
      spike.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy backends
# ---------------------------------------------------------------------------

_NVML_READY: Optional[bool] = None
_NVML = None

_PSUTIL_READY: Optional[bool] = None
_PSUTIL = None
_PROC = None  # psutil.Process(self) cached

# Throttle-reason bitmasks → human labels (NVML constants are stable)
_THROTTLE_BITS: list[tuple[int, str]] = [
    (0x0000000000000001, "gpu_idle"),
    (0x0000000000000002, "applications_clocks_setting"),
    (0x0000000000000004, "sw_power_cap"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal_slowdown"),
    (0x0000000000000040, "hw_thermal_slowdown"),
    (0x0000000000000080, "hw_power_brake_slowdown"),
    (0x0000000000000100, "display_clock_setting"),
]

# Cumulative-counter rate state (process-local, monotonic-ish)
# key -> (prev_value, prev_t)
_PREV: dict[str, tuple[float, float]] = {}


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


def _init_psutil() -> bool:
    global _PSUTIL_READY, _PSUTIL, _PROC
    if _PSUTIL_READY is not None:
        return _PSUTIL_READY
    try:
        import psutil
        _PSUTIL = psutil
        _PROC = psutil.Process(os.getpid())
        # Prime cpu_percent so the next call returns a real value
        _PROC.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        _PSUTIL_READY = True
    except Exception as e:
        logger.debug(f"psutil unavailable ({e}); CPU/host telemetry disabled")
        _PSUTIL_READY = False
    return _PSUTIL_READY


def _rate(key: str, value: float, now: float) -> float:
    """Convert a monotonic counter to a per-second rate.  First call → 0."""
    prev = _PREV.get(key)
    _PREV[key] = (value, now)
    if prev is None:
        return 0.0
    pv, pt = prev
    dt = now - pt
    if dt <= 0:
        return 0.0
    delta = value - pv
    if delta < 0:
        # Counter wrap or reset — best to drop this sample.
        return 0.0
    return delta / dt


# ---------------------------------------------------------------------------
# GPU sub-collector
# ---------------------------------------------------------------------------

def _gpu_metrics(now: float) -> dict[str, float]:
    if not _init_nvml():
        return {}
    out: dict[str, float] = {}
    nvml = _NVML
    try:
        n = nvml.nvmlDeviceGetCount()
    except Exception as e:
        logger.debug(f"nvmlDeviceGetCount failed: {e}")
        return out

    out["gpu/count"] = float(n)

    for i in range(n):
        try:
            h = nvml.nvmlDeviceGetHandleByIndex(i)
        except Exception:
            continue
        prefix = f"gpu/{i}/"

        # Util
        try:
            u = nvml.nvmlDeviceGetUtilizationRates(h)
            out[prefix + "util_pct"] = float(u.gpu)
            out[prefix + "mem_util_pct"] = float(u.memory)
        except Exception:
            pass

        # Memory
        try:
            m = nvml.nvmlDeviceGetMemoryInfo(h)
            out[prefix + "mem_used_mib"] = float(m.used) / (1024 * 1024)
            out[prefix + "mem_free_mib"] = float(m.free) / (1024 * 1024)
            out[prefix + "mem_total_mib"] = float(m.total) / (1024 * 1024)
            if m.total > 0:
                out[prefix + "mem_used_pct"] = 100.0 * float(m.used) / float(m.total)
        except Exception:
            pass

        # Temperature
        try:
            out[prefix + "temp_c"] = float(
                nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass

        # Power
        try:
            out[prefix + "power_w"] = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        except Exception:
            pass
        try:
            cap = nvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
            out[prefix + "power_limit_w"] = float(cap)
        except Exception:
            pass

        # Fan
        try:
            out[prefix + "fan_pct"] = float(nvml.nvmlDeviceGetFanSpeed(h))
        except Exception:
            pass

        # Clocks (MHz)
        try:
            out[prefix + "clock_sm_mhz"] = float(
                nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM)
            )
            out[prefix + "clock_mem_mhz"] = float(
                nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_MEM)
            )
        except Exception:
            pass

        # Performance state (P0=highest, P15=lowest)
        try:
            out[prefix + "perf_state"] = float(nvml.nvmlDeviceGetPerformanceState(h))
        except Exception:
            pass

        # PCIe throughput (KB/s) — these *are* rates, not counters
        try:
            tx = nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_TX_BYTES)
            rx = nvml.nvmlDeviceGetPcieThroughput(h, nvml.NVML_PCIE_UTIL_RX_BYTES)
            out[prefix + "pcie_tx_kbs"] = float(tx)
            out[prefix + "pcie_rx_kbs"] = float(rx)
        except Exception:
            pass

        # NVLink (sum across links) — counter values in bytes, exposed as MiB/s
        try:
            tx_total = 0
            rx_total = 0
            seen = False
            for link in range(0, 18):  # generous upper bound; break on error
                try:
                    state = nvml.nvmlDeviceGetNvLinkState(h, link)
                except Exception:
                    break
                if state != nvml.NVML_FEATURE_ENABLED:
                    continue
                seen = True
                try:
                    counter = nvml.nvmlDeviceGetNvLinkUtilizationCounter(h, link, 0)
                    tx_total += int(counter["tx"])
                    rx_total += int(counter["rx"])
                except Exception:
                    # Some driver/pynvml combos don't expose this counter.
                    pass
            if seen:
                # NVLink util counters report bytes — convert to MiB/s rate.
                tx_rate = _rate(
                    f"nvlink_tx_{i}", float(tx_total), now
                ) / (1024 * 1024)
                rx_rate = _rate(
                    f"nvlink_rx_{i}", float(rx_total), now
                ) / (1024 * 1024)
                out[prefix + "nvlink_tx_mibs"] = tx_rate
                out[prefix + "nvlink_rx_mibs"] = rx_rate
        except Exception:
            pass

        # ECC errors (volatile → since last reset; aggregate → device lifetime)
        try:
            sb_vol = nvml.nvmlDeviceGetTotalEccErrors(
                h, nvml.NVML_MEMORY_ERROR_TYPE_CORRECTED, nvml.NVML_VOLATILE_ECC
            )
            db_vol = nvml.nvmlDeviceGetTotalEccErrors(
                h, nvml.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nvml.NVML_VOLATILE_ECC
            )
            out[prefix + "ecc_sbe_volatile"] = float(sb_vol)
            out[prefix + "ecc_dbe_volatile"] = float(db_vol)
        except Exception:
            pass

        # Throttle reasons — emit a 0/1 gauge per known reason
        try:
            mask = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            for bit, label in _THROTTLE_BITS:
                out[prefix + f"throttle_{label}"] = 1.0 if (mask & bit) else 0.0
        except Exception:
            pass

    return out


# ---------------------------------------------------------------------------
# CPU / host sub-collector
# ---------------------------------------------------------------------------

def _cpu_metrics(now: float) -> dict[str, float]:
    if not _init_psutil():
        return {}
    psutil = _PSUTIL
    out: dict[str, float] = {}

    # Per-process
    try:
        out["proc/cpu_pct"] = float(_PROC.cpu_percent(interval=None))
    except Exception:
        pass
    try:
        m = _PROC.memory_info()
        out["proc/rss_mib"] = float(m.rss) / (1024 * 1024)
        out["proc/vms_mib"] = float(m.vms) / (1024 * 1024)
    except Exception:
        pass
    try:
        out["proc/num_threads"] = float(_PROC.num_threads())
    except Exception:
        pass
    try:
        out["proc/num_fds"] = float(_PROC.num_fds())
    except Exception:
        pass

    # System CPU
    try:
        out["sys/cpu_pct"] = float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    try:
        la1, la5, la15 = os.getloadavg()
        out["sys/loadavg_1m"] = float(la1)
        out["sys/loadavg_5m"] = float(la5)
        out["sys/loadavg_15m"] = float(la15)
    except Exception:
        pass

    # System memory
    try:
        vm = psutil.virtual_memory()
        out["sys/mem_used_mib"] = float(vm.used) / (1024 * 1024)
        out["sys/mem_available_mib"] = float(vm.available) / (1024 * 1024)
        out["sys/mem_total_mib"] = float(vm.total) / (1024 * 1024)
        out["sys/mem_used_pct"] = float(vm.percent)
    except Exception:
        pass
    try:
        sw = psutil.swap_memory()
        out["sys/swap_used_mib"] = float(sw.used) / (1024 * 1024)
        out["sys/swap_used_pct"] = float(sw.percent)
    except Exception:
        pass

    # Disk I/O — derive rates from the cumulative counters
    try:
        d = psutil.disk_io_counters()
        if d:
            out["sys/disk_read_mibs"] = (
                _rate("disk_read", float(d.read_bytes), now) / (1024 * 1024)
            )
            out["sys/disk_write_mibs"] = (
                _rate("disk_write", float(d.write_bytes), now) / (1024 * 1024)
            )
    except Exception:
        pass

    # Network I/O
    try:
        nio = psutil.net_io_counters()
        if nio:
            out["sys/net_recv_mibs"] = (
                _rate("net_recv", float(nio.bytes_recv), now) / (1024 * 1024)
            )
            out["sys/net_sent_mibs"] = (
                _rate("net_sent", float(nio.bytes_sent), now) / (1024 * 1024)
            )
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def system_stats(
    *,
    gpu: bool = True,
    cpu: bool = True,
) -> dict[str, float]:
    """
    Collect a flat dict of host + GPU metrics.

    Keys use ``/`` separators (e.g. ``gpu/0/util_pct``).  Pass the
    result straight to a Bhaskera logger; the Ray/Prometheus backend
    handles the name mangling internally.
    """
    now = time.time()
    out: dict[str, float] = {}
    if gpu:
        try:
            out.update(_gpu_metrics(now))
        except Exception as e:
            logger.debug(f"gpu_metrics failed: {e}")
    if cpu:
        try:
            out.update(_cpu_metrics(now))
        except Exception as e:
            logger.debug(f"cpu_metrics failed: {e}")
    return out


def cuda_memory_stats(device: Any = None) -> dict[str, float]:
    """
    Per-process CUDA memory — *PyTorch's* view, complementary to the
    NVML one above.  PyTorch knows about cached allocator state that
    NVML can't see.

    Keys:
        cuda/allocated_mib       - currently allocated (live tensors)
        cuda/reserved_mib        - reserved by the caching allocator
        cuda/max_allocated_mib   - peak since reset
        cuda/max_reserved_mib    - peak reserved since reset
    """
    out: dict[str, float] = {}
    try:
        import torch
        if not torch.cuda.is_available():
            return out
        if device is None:
            device = torch.cuda.current_device()
        out["cuda/allocated_mib"] = (
            torch.cuda.memory_allocated(device) / (1024 * 1024)
        )
        out["cuda/reserved_mib"] = (
            torch.cuda.memory_reserved(device) / (1024 * 1024)
        )
        out["cuda/max_allocated_mib"] = (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        )
        out["cuda/max_reserved_mib"] = (
            torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        )
    except Exception as e:
        logger.debug(f"cuda_memory_stats failed: {e}")
    return out
