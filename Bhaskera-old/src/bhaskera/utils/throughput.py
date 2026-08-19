"""
bhaskera.utils.throughput
=========================
Step-time, tokens/sec, samples/sec, and an MFU estimate for LLM
fine-tuning loops.

MFU follows the Chinchilla / PaLM / Karpathy convention exactly:

    flops_per_token = 6 * N  (2 fwd + 2 bwd_act + 2 bwd_weight)
                    + 12 * L * S * H  (attention; exact additive term)

    With activation checkpointing (one full forward recompute):
    flops_per_token = 8 * N  (+ 2 recomputed fwd)
                    + 16 * L * S * H  (attn recomputed too)

PEFT note:
    The dominant cost is still the full base-model forward + activation
    backward (needed to reach adapter weights).  The weight-gradient term
    for frozen params is skipped, but that is small relative to the full
    pass and NOT subtracted in the standard convention.  We use 6× / 8×
    for PEFT identical to full fine-tuning, matching PaLM / Karpathy.
    Pass ``params_for_flops`` as the *full* model param count.

MFU uses actual per-step wall-clock time (dt), not the EMA, so it
reflects true achieved hardware utilisation for that step.
A separate ``mfu_ema_pct`` key carries the EMA-smoothed version.

Fixes vs previous version
--------------------------
1. PEFT multiplier corrected: 4× → 6× (no checkpointing), 6× → 8× (+ckpt)
2. Checkpointing multiplier: PEFT+ckpt was 6× → now 8× (consistent)
3. MFU now computed from actual ``dt``, not from EMA ``ref_dt``
4. ``local_tps`` (instantaneous) and ``local_tps_ema`` are separate;
   MFU uses instantaneous, throughput reporting uses both
5. ``NameError`` guard: MFU block is inside the ``local_tokens > 0`` check
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional


class ThroughputTracker:
    """Lightweight tracker — call ``step()`` once per optimizer step.

    Parameters
    ----------
    params_for_flops:
        Full model parameter count (use base-model count even for PEFT).
    world_size:
        Total number of GPUs across all nodes.
    peak_flops_per_gpu:
        Theoretical peak FLOP/s for one GPU in the training dtype.
        Defaults to A100 BF16 (312 TFLOPS).
    window:
        Rolling window size for EMA step-time.
    warmup_steps:
        Steps to exclude from EMA (compilation / cache warmup).
    activation_checkpointing:
        Set True if gradient checkpointing is enabled.
    num_layers:
        Transformer layer count.  Required for exact attention FLOPs.
    hidden_size:
        Model hidden dimension.  Required for exact attention FLOPs.
    """

    def __init__(
        self,
        *,
        params_for_flops: int,
        world_size: int,
        peak_flops_per_gpu: float = 312e12,   # A100 BF16
        window: int = 50,
        warmup_steps: int = 5,
        # NOTE: is_peft removed — multiplier is identical to full FT
        # (kept as silent kwarg for backward compat so call-sites don't break)
        is_peft: bool = False,                # retained for API compat only
        activation_checkpointing: bool = False,
        num_layers: int = 0,
        hidden_size: int = 0,
    ) -> None:
        self._params = max(1, int(params_for_flops))
        self._world  = max(1, int(world_size))
        self._peak   = max(1.0, float(peak_flops_per_gpu))
        self._window = max(1, int(window))
        self._warmup = max(0, int(warmup_steps))

        self._checkpointing = activation_checkpointing
        self._num_layers    = num_layers
        self._hidden_size   = hidden_size

        self._step_times: deque[float] = deque(maxlen=self._window)
        self._last_t: Optional[float] = None
        self._steps_seen = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _flops_per_token(self) -> float:
        """
        Compute FLOPs per token following the Chinchilla/PaLM formula.

        Base term (dense matmuls, forward + backward):
            Standard:              6 × N
            + activation ckpt:     8 × N   (+2 for recomputed forward)

        Attention term (exact; additive):
            Standard:             12 × L × S × H
            + activation ckpt:    16 × L × S × H  (+4 recomputed attn)

        This applies to both full FT and PEFT because the full base-model
        forward and activation-backward are always executed.
        """
        base_mult = 8.0 if self._checkpointing else 6.0
        flops = base_mult * self._params

        if self._num_layers > 0 and self._hidden_size > 0:
            # seq_len is passed at call time; placeholder here
            pass   # see step() — seq_len is injected there

        return flops   # attention term added in step() where seq_len is known

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_step_clock(self) -> None:
        """Call right before the first forward of a new step."""
        self._last_t = time.perf_counter()

    def step(
        self,
        *,
        local_tokens_in_step: int,   # tokens processed by ONE GPU this step
        local_samples_in_step: int,  # samples processed by ONE GPU this step
        seq_len: int,
    ) -> dict[str, float]:
        """
        Close out one optimizer step and return derived metrics.

        Returns
        -------
        dict with keys::

            throughput/step_time_s              — wall time this step (s)
            throughput/step_time_ema_s          — EMA of step times (s)
            throughput/tokens_per_sec_per_gpu   — instantaneous, single GPU
            throughput/tokens_per_sec_per_gpu_ema  — EMA-smoothed, single GPU
            throughput/tokens_per_sec_global    — instantaneous × world_size
            throughput/samples_per_sec_global   — instantaneous × world_size
            throughput/mfu_pct                  — true MFU (uses actual dt)
            throughput/mfu_ema_pct              — EMA-smoothed MFU
            throughput/total_steps
        """
        now = time.perf_counter()
        out: dict[str, float] = {}
        self._steps_seen += 1
        out["throughput/total_steps"] = float(self._steps_seen)

        if self._last_t is None:
            self._last_t = now
            return out

        # Actual wall-clock time for this step
        dt = now - self._last_t
        self._last_t = now

        if dt <= 0:
            return out

        out["throughput/step_time_s"] = dt

        # --- EMA step time -------------------------------------------
        if self._steps_seen > self._warmup:
            self._step_times.append(dt)

        if self._step_times:
            ema_dt = sum(self._step_times) / len(self._step_times)
            out["throughput/step_time_ema_s"] = ema_dt
        else:
            ema_dt = dt   # before warmup ends, fall back to actual

        # --- FLOPs per token (base term + exact attention) ------------
        base_mult = 8.0 if self._checkpointing else 6.0
        flops_per_token = base_mult * self._params

        if self._num_layers > 0 and self._hidden_size > 0:
            # Attention: 4 ops × Q×K, QK×V, softmax ≈ 12LSH standard
            #            + 4LSH for recomputed fwd attn if checkpointing
            att_mult = 16.0 if self._checkpointing else 12.0
            flops_per_token += att_mult * self._num_layers * seq_len * self._hidden_size

        # --- Throughput & MFU ----------------------------------------
        if local_tokens_in_step > 0:
            # Instantaneous (uses actual dt) — this is what MFU is based on
            local_tps_inst = local_tokens_in_step / dt
            out["throughput/tokens_per_sec_per_gpu"]     = local_tps_inst
            out["throughput/tokens_per_sec_global"]      = local_tps_inst * self._world

            # EMA-smoothed version (for dashboards / less noisy logging)
            local_tps_ema = local_tokens_in_step / ema_dt
            out["throughput/tokens_per_sec_per_gpu_ema"] = local_tps_ema

            # FIX 3 & 4: MFU uses instantaneous tps, not EMA
            achieved_flops = flops_per_token * local_tps_inst
            out["throughput/mfu_pct"] = 100.0 * (achieved_flops / self._peak)

            # Bonus: EMA-smoothed MFU (less noisy, good for steady-state)
            achieved_flops_ema = flops_per_token * local_tps_ema
            out["throughput/mfu_ema_pct"] = 100.0 * (achieved_flops_ema / self._peak)

        if local_samples_in_step > 0:
            local_sps_inst = local_samples_in_step / dt
            out["throughput/samples_per_sec_global"] = local_sps_inst * self._world

        return out
