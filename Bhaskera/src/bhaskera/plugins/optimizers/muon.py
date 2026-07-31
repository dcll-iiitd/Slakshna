# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Muon optimizer with optional 8-bit or 4-bit quantized momentum buffer.

Design notes (Phase 1 — pure-Python, no new C++ kernels):
  - Persistent state is a single momentum buffer quantized either to
    uint8 (blockwise dynamic 8-bit, blocksize=256) or to packed NF4/FP4
    (4-bit, blocksize=64), matching the Optimizer1State layout used by SGD8bit.
  - Per step, per parameter (or parameter batch):
      1. Dequantize m → fp32 transient.
      2. Momentum update: m = β·m + g; Nesterov input u = g + β·m (or u = m).
      3. Requantize m back into state1/absmax1.
      4. Orthogonalize u via Newton-Schulz in bf16.
      5. Apply weight decay + param update.
  - An `orthogonalize_fn` hook accepts any callable X → X. If the
    `gram_newton_schulz` package is importable, GramNewtonSchulz (torch
    backend; CuTeDSL kernels only on sm90/sm100 with quack installed) is used
    by default; otherwise a pure-torch standard NS fallback.
  - Same-shape parameters are processed in chunks of `ns_chunk_size` through a
    pre-allocated bf16 buffer to bound the transient memory of the batched NS.
  - Parameters with ndim < 2 should be placed in a separate AdamW/SGD group.

Memory footprint (approximate, per trainable element):
  - 32-bit:  4 bytes  (fp32 momentum)
  -  8-bit:  1 byte   (uint8 blockwise dynamic, blocksize=256)
  -  4-bit:  0.5 byte (NF4/FP4 packed, blocksize=64) + absmax overhead ≈ 0.52 bytes total
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import chain
import math
from typing import Optional

import torch
from torch import Tensor

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import MockArgs, Optimizer8bit

# ---------------------------------------------------------------------------
# Optional DTensor / distributed imports (FSDP2 support)
# ---------------------------------------------------------------------------
try:
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, Replicate, Shard

    _DTENSOR_AVAILABLE = True
except ImportError:
    dist = None  # type: ignore[assignment]
    DTensor = None  # type: ignore[assignment]
    Shard = None  # type: ignore[assignment]
    Replicate = None  # type: ignore[assignment]
    _DTENSOR_AVAILABLE = False

try:
    from bitsandbytes.backends.triton.kernels_muon import (
        muon_momentum_4bit_fused,
        muon_momentum_8bit_fused,
        muon_momentum_nvfp4_fused,
    )
except ImportError:  # triton not available
    muon_momentum_8bit_fused = None
    muon_momentum_4bit_fused = None
    muon_momentum_nvfp4_fused = None

# ---------------------------------------------------------------------------
# Polar Express coefficients (https://arxiv.org/pdf/2505.16932)
# ---------------------------------------------------------------------------
_UNMODIFIED_POLAR_EXPRESS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
]
_SAFETY = 1.05
POLAR_EXPRESS_COEFFICIENTS: list[tuple[float, float, float]] = [
    (a / _SAFETY, b / _SAFETY**3, c / _SAFETY**5) for (a, b, c) in _UNMODIFIED_POLAR_EXPRESS
]


# ---------------------------------------------------------------------------
# Pure-PyTorch Newton-Schulz (standard, operates on the full matrix)
# ---------------------------------------------------------------------------
def _standard_newton_schulz(
    X: Tensor,
    coefficients: list[tuple[float, float, float]] = POLAR_EXPRESS_COEFFICIENTS,
    eps: float = 1e-7,
) -> Tensor:
    """
    Standard Newton-Schulz orthogonalization (5 iterations by default).

    Operates in bf16 after Frobenius normalization, matching the gram-repo design.
    Works on a single 2-D matrix or a batch with leading batch dims.

    Args:
        X: (..., m, n) tensor
        coefficients: list of (a, b, c) triples; one per iteration
        eps: small value for safe Frobenius normalization

    Returns:
        Orthogonalized tensor of the same shape as X
    """
    orig_dtype = X.dtype
    orig_shape = X.shape

    # baddbmm requires 3-D batched tensors
    if X.ndim == 2:
        X = X.unsqueeze(0)
    elif X.ndim > 3:
        X = X.reshape(-1, *X.shape[-2:])

    X = X.to(torch.bfloat16)

    # Frobenius normalize
    norm = X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    X = X / norm

    # Determine if we need to transpose for the "tall" case (m > n)
    transposed = False
    if X.shape[-2] > X.shape[-1]:
        X = X.mT
        transposed = True

    X = X.contiguous()
    for a, b, c in coefficients:
        A = X @ X.mT
        # B = b*A + c*(A @ A);  X = a*X + B @ X   — two fused kernels keep the
        # live temporaries to one Gram-sized and one X-sized tensor.
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a)

    if transposed:
        X = X.mT

    return X.to(orig_dtype).reshape(orig_shape)


# ---------------------------------------------------------------------------
# NVFP4 Python eager helpers (CPU / Triton-unavailable fallback)
# ---------------------------------------------------------------------------
# NVFP4 magnitude values normalised to [-1, 1] (indexed by mag code 0-7):
_NVFP4_MAG_VALUES: list[float] = [
    0.0,
    0.5 / 6,
    1.0 / 6,
    1.5 / 6,
    2.0 / 6,
    3.0 / 6,
    4.0 / 6,
    1.0,
]
# Quantise thresholds: midpoints between adjacent magnitudes (in [0, 1]):
_NVFP4_THRESHOLDS: list[float] = [
    0.25 / 6,  # 0 ↔ 1
    0.75 / 6,  # 1 ↔ 2
    1.25 / 6,  # 2 ↔ 3
    1.75 / 6,  # 3 ↔ 4
    2.5 / 6,  # 4 ↔ 5
    3.5 / 6,  # 5 ↔ 6
    5.0 / 6,  # 6 ↔ 7
]


def _nvfp4_dequantize_eager(
    packed: Tensor,  # uint8, flat, (ceil(n/2),)
    absmax: Tensor,  # float32, (ceil(n/blocksize),)
    n: int,
    blocksize: int,
    device,
) -> Tensor:
    """Pure-Python NVFP4 dequantize (CPU / fallback).  Returns float32 (n,)."""
    code = torch.tensor(_NVFP4_MAG_VALUES, dtype=torch.float32, device=device)
    n_paired = (n + 1) // 2
    pairs_per_block = blocksize // 2

    code_hi = (packed >> 4).long()  # (n_paired,) — first element of each pair
    code_lo = (packed & 0xF).long()  # (n_paired,) — second element

    sign_hi = torch.where(code_hi >= 8, -1.0, 1.0)
    sign_lo = torch.where(code_lo >= 8, -1.0, 1.0)
    val_hi = sign_hi * code[code_hi & 0x7]  # (n_paired,)
    val_lo = sign_lo * code[code_lo & 0x7]  # (n_paired,)

    pair_idx = torch.arange(n_paired, device=device)
    blk_idx = (pair_idx // pairs_per_block).clamp(max=absmax.numel() - 1)
    s = absmax[blk_idx]  # (n_paired,)

    out = torch.empty(n, dtype=torch.float32, device=device)
    out[0::2] = val_hi * s
    n_lo = n_paired if n % 2 == 0 else n_paired - 1
    if n_lo > 0:
        out[1::2] = (val_lo * s)[:n_lo]
    return out


def _nvfp4_quantize_eager(
    m: Tensor,  # float32, flat (n,)
    packed: Tensor,  # uint8, flat (ceil(n/2),) — modified in place
    absmax: Tensor,  # float32 (ceil(n/blocksize),) — modified in place
    blocksize: int,
) -> None:
    """Pure-Python NVFP4 quantise (CPU / fallback).  Updates packed and absmax."""
    n = m.numel()
    n_paired = (n + 1) // 2
    device = m.device
    pairs_per_block = blocksize // 2

    # Pad to even length for pair-processing
    m_pad = m if n % 2 == 0 else torch.cat([m, m.new_zeros(1)])

    # Per-block absmax. Pad the final partial block with zeros (zeros never
    # raise the block's abs-max) so reshape works for any n.
    n_blocks = (n + blocksize - 1) // blocksize
    block_pad = n_blocks * blocksize - n
    m_for_blocks = m if block_pad == 0 else torch.cat([m, m.new_zeros(block_pad)])
    absmax_new = m_for_blocks.reshape(n_blocks, blocksize).abs().amax(dim=1)  # (n_blocks,)
    absmax.copy_(absmax_new)

    # Per-pair scale
    pair_idx = torch.arange(n_paired, device=device)
    blk_idx = (pair_idx // pairs_per_block).clamp(max=n_blocks - 1)
    s = absmax_new[blk_idx].clamp(min=1e-12)  # (n_paired,)

    # Normalise and quantise (sign-magnitude)
    m_hi = m_pad[0::2] / s  # (n_paired,) — even elements ∈ [-1, 1]
    m_lo = m_pad[1::2] / s  # (n_paired,)

    def _quant_mag(x: Tensor) -> Tensor:
        """Map abs(x) ∈ [0,1] to magnitude code 0-7."""
        x_abs = x.abs().clamp(0.0, 1.0)
        code = torch.zeros_like(x_abs, dtype=torch.long)
        for thresh in _NVFP4_THRESHOLDS:
            code += (x_abs >= thresh).long()
        return code

    sign_hi = (m_hi < 0).long() * 8
    sign_lo = (m_lo < 0).long() * 8
    codes_hi = (sign_hi | _quant_mag(m_hi)).to(torch.uint8)  # (n_paired,)
    codes_lo = (sign_lo | _quant_mag(m_lo)).to(torch.uint8)  # (n_paired,)

    # Pack: high nibble = first element (even index), low nibble = second (odd)
    packed.copy_(((codes_hi << 4) | (codes_lo & 0xF)).to(torch.uint8))


# ---------------------------------------------------------------------------
# NVFP4 x sm100: Newton-Schulz using Blackwell FP4 tensor cores for X @ X.T
# ---------------------------------------------------------------------------


def _to_nvfp4_with_scales(X: Tensor, block_size: int = 16) -> tuple[Tensor, Tensor]:
    """Pack (M, N) bf16/fp32 tensor into NVFP4 with 1xblock_size FP8 block scales.

    Returns:
        X_fp4:  (M, N//2)      dtype=torch.float4_e2m1fn_x2
        scales: (M, N//block_size)  dtype=torch.float8_e4m3fn

    Packing follows the same convention as torch._bfloat16_to_float4_e2m1fn_x2:
      packed_byte[i] = (code[2*i+1] << 4) | (code[2*i] & 0xF)
      i.e. the odd-index element goes to the high nibble.

    Requires N to be a multiple of block_size (= 16 for sm100 BlockWise1x16).
    """
    X = X.float()
    M, N = X.shape
    assert N % block_size == 0, f"_to_nvfp4_with_scales: N={N} must be divisible by block_size={block_size}"
    N_blocks = N // block_size

    # Per-block absmax → FP8 scales
    X_blk = X.reshape(M, N_blocks, block_size)  # (M, Nb, B)
    absmax = X_blk.abs().amax(dim=-1).clamp(min=1e-12)  # (M, Nb)
    FP4_MAX, FP8_MAX = 6.0, 448.0
    scale_fp8 = (absmax / FP4_MAX).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)  # (M, Nb)

    # Normalize to [-FP4_MAX, FP4_MAX] then quantize
    X_norm = (X_blk / absmax.unsqueeze(-1) * FP4_MAX).clamp(-FP4_MAX, FP4_MAX)  # (M, Nb, B)
    X_flat = X_norm.reshape(M, N)  # (M, N)

    x_abs = X_flat.abs()
    sign_bit = (X_flat < 0).to(torch.int32) * 8  # (M, N)
    mag = (x_abs >= 5.0).long() * 7
    for the, lv in [(3.5, 6), (2.5, 5), (1.75, 4), (1.25, 3), (0.75, 2), (0.25, 1)]:
        mag = torch.where((x_abs >= the) & (mag == 0), torch.full_like(mag, lv), mag)
    # Compact equivalent:
    mag = (
        ((x_abs >= 5.0).long()) * 7
        + ((x_abs >= 3.5) & (x_abs < 5.0)).long() * 6
        + ((x_abs >= 2.5) & (x_abs < 3.5)).long() * 5
        + ((x_abs >= 1.75) & (x_abs < 2.5)).long() * 4
        + ((x_abs >= 1.25) & (x_abs < 1.75)).long() * 3
        + ((x_abs >= 0.75) & (x_abs < 1.25)).long() * 2
        + ((x_abs >= 0.25) & (x_abs < 0.75)).long() * 1
    )
    codes = (sign_bit | mag).to(torch.uint8)  # (M, N), values 0-15

    # Pack pairs (odd → high nibble, even → low nibble — matches pack_uint4)
    packed = (codes[:, 1::2].to(torch.int32) << 4 | codes[:, ::2].to(torch.int32) & 0xF).to(torch.uint8)  # (M, N//2)
    X_fp4 = packed.view(torch.float4_e2m1fn_x2)  # (M, N//2)
    return X_fp4, scale_fp8


def _make_sm100_nvfp4_ns_fn(
    coefficients: list[tuple[float, float, float]] = POLAR_EXPRESS_COEFFICIENTS,
    eps: float = 1e-7,
) -> Callable[[Tensor], Tensor]:
    """Return a Newton-Schulz function that uses Blackwell sm100 FP4 tensor
    cores for the X @ X.T GEMM (the most expensive step).

    Only the Gram matrix computation uses FP4 (via torch._scaled_grouped_mm_v2
    with float4_e2m1fn_x2 inputs and float8_e4m3fn block scales, BlockWise1x16,
    SWIZZLE_32_4_4).  All other operations (A@A, polynomial, X update) stay in
    BF16 for numerical stability.  Iteration count and coefficients are the
    same as _standard_newton_schulz.

    Falls back to _standard_newton_schulz if:
    - torch._scaled_grouped_mm_v2 is not available, or
    - N (or M after potential transpose) is not a multiple of 16, or
    - not running on CUDA.
    """
    try:
        from torch.nn.functional import ScalingType, SwizzleType

        _sgmm = torch._scaled_grouped_mm_v2
    except (ImportError, AttributeError):
        return _default_orthogonalize_fn

    def _fp4_gram(X_single: Tensor) -> Tensor:
        """Compute X @ X.T for one (m, n) matrix using sm100 FP4 GEMM → bf16."""
        _m, n = X_single.shape
        if n % 16 != 0:
            # Dimensions not suitable for BlockWise1x16; use bf16 fallback.
            return X_single @ X_single.mT
        X_fp4, scale_X = _to_nvfp4_with_scales(X_single.contiguous(), block_size=16)
        # mat2 = X.T packed as mat2: transpose the packed tensor (each byte
        # still encodes the correct pair of elements for the contracted K dim).
        X_T_fp4 = X_fp4.t().contiguous()  # (n//2, m)
        scale_X_T = scale_X.t().contiguous()  # (n//16, m)
        return _sgmm(
            X_fp4,
            X_T_fp4,
            [scale_X],
            [ScalingType.BlockWise1x16],
            [SwizzleType.SWIZZLE_32_4_4],
            [scale_X_T],
            [ScalingType.BlockWise1x16],
            [SwizzleType.SWIZZLE_32_4_4],
            None,
            torch.bfloat16,
        )

    def _nvfp4_ns(X: Tensor) -> Tensor:
        if not X.is_cuda:
            return _standard_newton_schulz(X, coefficients, eps)

        orig_dtype = X.dtype
        orig_shape = X.shape

        if X.ndim == 2:
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            X = X.reshape(-1, *X.shape[-2:])

        X = X.to(torch.bfloat16)
        norm = X.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
        X = X / norm

        transposed = False
        if X.shape[-2] > X.shape[-1]:
            X = X.mT.contiguous()
            transposed = True

        B, _m, _n = X.shape

        for a, b, c in coefficients:
            # FP4 GEMM for X @ X.T (one call per batch element)
            A = torch.stack([_fp4_gram(X[bi]) for bi in range(B)], dim=0)  # (B, m, m)
            # BF16 polynomial: B = b*A + c*(A@A)
            Bpoly = torch.baddbmm(A, A, A, beta=b, alpha=c)
            # BF16 update: X = a*X + Bpoly@X
            X = torch.baddbmm(X, Bpoly, X, beta=a)

        if transposed:
            X = X.mT.contiguous()

        return X.to(orig_dtype).reshape(orig_shape)

    return _nvfp4_ns


def _default_orthogonalize_fn(
    X: Tensor,
) -> Tensor:
    """Default orthogonalize function using pure-PyTorch standard NS."""
    return _standard_newton_schulz(X)


_DEFAULT_ORTHOGONALIZE_FN: Optional[Callable[[Tensor], Tensor]] = None


def newton_schulz(X: Tensor) -> Tensor:
    """Backward-compatible public Newton-Schulz helper.

    The optimizer feeds Newton-Schulz with bf16 momentum directions and uses
    GramNewtonSchulz when available; this helper mirrors that default path for
    tests and external callers that compare against Muon step-by-step.
    """
    return _get_default_orthogonalize_fn()(X.to(torch.bfloat16))


def _make_default_orthogonalize_fn() -> Callable[[Tensor], Tensor]:
    """
    Prefer GramNewtonSchulz (Dao-AILab) when the package is importable.

    The Gram iteration runs on the small nxn Gram matrix instead of the full
    mxn matrix (~2x fewer FLOPs and far smaller transients for rectangular
    weights) and works on any GPU through its torch backend. The CuTeDSL
    symmetric-GEMM kernels are only enabled on sm90/sm100 (H100/B200) when
    quack is installed; consumer/workstation Blackwell (sm120) is not a
    supported kernel target.

    compile_kwargs=None: reduce-overhead mode uses CUDA graphs whose memory
    pools inflate reserved memory across the varying chunk shapes seen here.
    """
    try:
        from gram_newton_schulz import GramNewtonSchulz
    except ImportError:
        return _default_orthogonalize_fn

    use_kernels = False
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) in ((9, 0), (10, 0), (10, 3)):
            try:
                import quack  # noqa: F401

                use_kernels = True
            except ImportError:
                use_kernels = False

    gns = GramNewtonSchulz(ns_use_kernels=use_kernels, compile_kwargs=None)

    if not use_kernels:
        # No CuTeDSL kernels on this GPU: attach our Triton symmetric-GEMM
        # backend (lower-triangle tiles only + mirrored store, fused
        # beta*C + alpha*A@B epilogue). Non-symmetric products stay on cuBLAS.
        try:
            from bitsandbytes.backends.triton.kernels_sym_gemm import make_triton_sym_backend

            gns._kernel_backend = make_triton_sym_backend()
        except ImportError:
            pass

    return gns.__call__


def _get_default_orthogonalize_fn() -> Callable[[Tensor], Tensor]:
    global _DEFAULT_ORTHOGONALIZE_FN
    if _DEFAULT_ORTHOGONALIZE_FN is None:
        _DEFAULT_ORTHOGONALIZE_FN = _make_default_orthogonalize_fn()
    return _DEFAULT_ORTHOGONALIZE_FN


# ---------------------------------------------------------------------------
# LR adjustment helpers (ported from gram-newton-schulz)
# ---------------------------------------------------------------------------
def _adjust_lr_rms_norm(lr: float, shape: tuple[int, ...]) -> float:
    fan_out, fan_in = shape[-2], shape[-1]
    return lr * 0.2 * math.sqrt(max(fan_out, fan_in))


def _adjust_lr_spectral_norm(lr: float, shape: tuple[int, ...]) -> float:
    fan_out, fan_in = shape[-2], shape[-1]
    return lr * math.sqrt(fan_out / fan_in)


_ADJUST_LR_MAP = {
    "rms_norm": _adjust_lr_rms_norm,
    "spectral_norm": _adjust_lr_spectral_norm,
    None: lambda lr, shape: lr,
}


def _resolve_adjust_lr(adjust_lr) -> Callable[[float, tuple[int, ...]], float]:
    if adjust_lr is None:
        return lambda lr, shape: lr
    if isinstance(adjust_lr, str):
        if adjust_lr not in _ADJUST_LR_MAP:
            raise ValueError(
                f"Invalid adjust_lr: {adjust_lr!r}. Must be 'rms_norm', 'spectral_norm', None, or a callable."
            )
        return _ADJUST_LR_MAP[adjust_lr]
    if callable(adjust_lr):
        return adjust_lr
    raise TypeError(f"adjust_lr must be str, None, or callable; got {type(adjust_lr)}")


# ---------------------------------------------------------------------------
# FSDP2 / DTensor distributed helpers — Layer A (pure, no collectives)
# ---------------------------------------------------------------------------


def _is_sharded(p: Tensor) -> bool:
    """True iff p is a DTensor with at least one Shard placement (FSDP2).

    False for plain tensors and fully-Replicate DTensors.  Never raises.
    """
    if not _DTENSOR_AVAILABLE or DTensor is None:
        return False
    return isinstance(p, DTensor) and any(isinstance(pl, Shard) for pl in p.placements)


def _assert_supported_layout(p: Tensor) -> None:
    """No-op for: plain tensor, Replicate DTensor, Shard DTensor with global ndim>=2.

    Raises NotImplementedError for unsupported layouts.  Error message MUST
    contain the bracketed token listed below so tests can match on it:

      - FSDP1 FlatParameter                   -> '[FSDP1]'
      - DeepSpeed ZeRO-3 partitioned param    -> '[ZeRO-3]'
      - Sharded param with global ndim < 2    -> '[ndim]'
      - is_paged=True + sharded               -> '[paged]'  (checked separately)

    Message also contains the literal 'fully_shard' pointing users to FSDP2.
    """
    # FSDP1 FlatParameter: 1-D flat blob of many params concatenated.
    if type(p).__name__ == "FlatParameter":
        raise NotImplementedError(
            "[FSDP1] FSDP1 FlatParameter is not supported by Muon. "
            "Replace model wrapping with fully_shard (FSDP2 / DTensor)."
        )
    # DeepSpeed ZeRO-3 partitioned params carry ds_* attributes.
    if hasattr(p, "ds_shape") or hasattr(p, "ds_numel") or hasattr(p, "ds_id"):
        raise NotImplementedError(
            "[ZeRO-3] DeepSpeed ZeRO-3 partitioned parameters are not supported "
            "by Muon.  Use fully_shard (FSDP2 / DTensor) instead."
        )
    # DTensor with global ndim < 2 (e.g. a sharded bias — should be in an AdamW group).
    if _DTENSOR_AVAILABLE and DTensor is not None and isinstance(p, DTensor):
        if p.ndim < 2:
            raise NotImplementedError(
                "[ndim] Muon requires parameters with global ndim >= 2 under FSDP2. "
                "Place 1-D params (biases, norms) in a separate AdamW group. "
                "Use fully_shard (FSDP2 / DTensor) only for 2-D+ parameters."
            )


def _global_matrix_shape(p: Tensor) -> tuple[int, int]:
    """Return the (fan_out, fan_in) 2-D shape used for NS, derived from the GLOBAL shape.

    For a DTensor p.shape is the global shape; for a plain Tensor it is the regular
    shape.  Applies the same collapse rule as _split_tensor_for_orthogonalization:
      2-D -> as-is
      n-D -> (shape[0], prod(shape[1:]))
    Pure; does not gather or communicate.
    """
    shape = p.shape  # global shape for DTensor, local shape for plain Tensor
    if len(shape) == 2:
        return (shape[0], shape[1])
    fan_in = 1
    for s in shape[1:]:
        fan_in *= s
    return (shape[0], fan_in)


def _assign_owners(costs: list[int], world_size: int) -> list[int]:
    """Size-aware greedy LPT (Longest Processing Time) owner assignment.

    costs[i] = NS cost proxy for param i; caller passes rows*cols*min(rows,cols)
    from _global_matrix_shape.  Returns owner rank per param in [0, world_size).

    Contract:
      - Deterministic: identical output for identical (costs, world_size).
      - Tie-break: ascending rank index, stable in input order.
      - len(output) == len(costs); every element in range(world_size).
    """
    if world_size <= 1 or not costs:
        return [0] * len(costs)
    load = [0] * world_size
    result = [0] * len(costs)
    # Process in descending cost order (LPT); stable sort on original index for ties.
    indexed = sorted(range(len(costs)), key=lambda i: (-costs[i], i))
    for i in indexed:
        # Assign to the least-loaded rank; tie-break by ascending rank index.
        min_rank = min(range(world_size), key=lambda r: (load[r], r))
        result[i] = min_rank
        load[min_rank] += costs[i]
    return result


# ---------------------------------------------------------------------------
# FSDP2 / DTensor distributed helpers — collective primitives
# ---------------------------------------------------------------------------


def _get_param_pg(p: Tensor):
    """Return the ProcessGroup for the first Shard dimension of a DTensor param."""
    mesh = p.device_mesh
    for dim, pl in enumerate(p.placements):
        if isinstance(pl, Shard):
            return mesh.get_group(dim)
    return mesh.get_group(0)


def _param_shard_world_size(p: Tensor) -> int:
    if not _is_sharded(p) or dist is None or not dist.is_initialized():
        return 1
    return dist.get_world_size(_get_param_pg(p))


def _gather_to_owner(u_local: Tensor, p: Tensor, owner: int) -> Optional[Tensor]:
    """Slice 2: Gather the local shard u_local from all ranks to the owner rank.

    Uses explicit dist.gather over the param's process group.  Returns the
    reconstructed full (fan_out, fan_in) tensor on the owner, None elsewhere.
    For non-sharded params returns u_local unchanged (trivially on the owner).
    """
    if not _is_sharded(p):
        return u_local
    pg = _get_param_pg(p)
    rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)

    local_flat = u_local.contiguous().view(-1)
    local_size = local_flat.numel()

    if rank == owner:
        gather_list = [
            torch.empty(local_size, dtype=local_flat.dtype, device=local_flat.device) for _ in range(world_size)
        ]
        dist.gather(local_flat, gather_list, dst=owner, group=pg)
        full_flat = torch.cat(gather_list, dim=0)
        # Strip FSDP2 per-shard padding using global shape (unpadded).
        global_shape = p.shape
        n_global = 1
        for s in global_shape:
            n_global *= s
        return full_flat[:n_global].reshape(global_shape)
    else:
        dist.gather(local_flat, None, dst=owner, group=pg)
        return None


def _broadcast_and_reshard(update_full: Optional[Tensor], p: Tensor, owner: int, local_shape: tuple) -> Tensor:
    """Slice 2: Broadcast the full update from the owner to all ranks, then reshard.

    Owner provides update_full; non-owners receive it via broadcast, then every
    rank extracts its local shard using DTensor redistribute.
    """
    pg = _get_param_pg(p)
    rank = dist.get_rank(pg)

    if rank == owner:
        buf = update_full.contiguous()
    else:
        # Allocate a buffer with the global shape to receive the broadcast.
        global_shape = p.shape
        buf = torch.empty(
            global_shape,
            dtype=update_full.dtype if update_full is not None else torch.float32,
            device=p.to_local().device,
        )

    dist.broadcast(buf, src=owner, group=pg)

    # Reshard the global update to this rank's local slice.
    replicate_placements = [Replicate() for _ in p.placements]
    update_dt = DTensor.from_local(buf, p.device_mesh, replicate_placements, run_check=False)
    return update_dt.redistribute(p.device_mesh, p.placements).to_local()


# ---------------------------------------------------------------------------
# DTensor state helpers — wrap / unwrap optimizer state buffers
# ---------------------------------------------------------------------------


def _as_state_dtensor(local: Tensor, p: Tensor) -> Tensor:
    """Wrap a local state buffer as a DTensor matching *p*'s sharding.

    Storing state1/absmax1 as live DTensors lets PyTorch DCP
    (``get_optimizer_state_dict`` / ``dcp.save``) gather and reshard them
    across world sizes without any custom ``state_dict`` override.

    For non-sharded params the local tensor is returned unchanged.
    """
    if not _is_sharded(p):
        return local
    return DTensor.from_local(local, p.device_mesh, p.placements, run_check=False)


def _local_tensor(t: Tensor) -> Tensor:
    """Return a local plain tensor for either a Tensor or DTensor."""
    if DTensor is not None and isinstance(t, DTensor):
        return t.to_local()
    return t


def _state_local(t: Tensor) -> Tensor:
    """Return a local view of a state buffer, preserving in-place DTensor writes."""
    return _local_tensor(t)


# ---------------------------------------------------------------------------
# Parameter batching (group same-shape params for batched NS)
# ---------------------------------------------------------------------------
def _create_param_batches(params: list[Tensor]) -> list[list[Tensor]]:
    """Group params by (global_shape, dtype, device) for a single batched NS call.

    For DTensor params (FSDP2) the key uses the global shape (p.shape) and the
    local device (p.to_local().device).  Plain tensors are unchanged.
    """
    groups: dict = defaultdict(list)
    for p in params:
        # p.shape is global shape for DTensor, regular shape for plain Tensor.
        shape = p.shape
        dtype = p.dtype
        device = p.to_local().device if _is_sharded(p) else p.device
        groups[(shape, dtype, device)].append(p)
    batches = []
    for group in groups.values():

        def _ptr(q: Tensor) -> int:
            return q.to_local().data_ptr() if _is_sharded(q) else q.data_ptr()

        group.sort(key=_ptr)
        batches.append(group)
    return batches


def _split_tensor_for_orthogonalization(x: Tensor):
    """Return 2-D matrices and enough metadata to rebuild x."""
    if x.ndim == 2:
        return [x], ("matrix", x.shape)
    if x.ndim == 3:
        return [x[i] for i in range(x.shape[0])], ("batch3d", x.shape)
    return [x.reshape(x.shape[0], -1)], ("flatten", x.shape)


def _reconstruct_tensor_from_matrices(spec, matrices: list[Tensor]) -> Tensor:
    kind, shape = spec
    if kind == "matrix":
        return matrices[0]
    if kind == "batch3d":
        return torch.stack(matrices, dim=0)
    if kind == "flatten":
        return matrices[0].reshape(shape)
    raise RuntimeError(f"Unknown Muon reconstruction spec: {kind}")


def _validate_param_split_fn(param_split_fn: Callable, x: Tensor, splits) -> list[Tensor]:
    fn_name = getattr(param_split_fn, "__name__", repr(param_split_fn))
    if not isinstance(splits, (list, tuple)) or len(splits) == 0:
        raise ValueError(f"param_split_fn ({fn_name}) must return a non-empty list/tuple of tensors")

    split_tensors = list(splits)
    for split in split_tensors:
        if not isinstance(split, torch.Tensor):
            raise TypeError(f"param_split_fn ({fn_name}) returned a non-tensor value: {type(split)}")
        if split.ndim != x.ndim:
            raise ValueError(f"param_split_fn ({fn_name}) must preserve ndim. Input: {x.ndim}D, output: {split.ndim}D")
        if split.ndim < 2:
            raise ValueError(f"param_split_fn ({fn_name}) returned a tensor with fewer than 2 dimensions")
        if x.ndim == 3 and split.shape[0] != x.shape[0]:
            raise ValueError(
                f"param_split_fn ({fn_name}) for 3D tensors must preserve the first dimension. "
                f"Input shape: {tuple(x.shape)}, output shape: {tuple(split.shape)}"
            )
    return split_tensors


def _prepare_orthogonalization_inputs(
    ns_inputs: list[Tensor],
    param_split_fn: Optional[Callable],
):
    matrices_by_shape: dict = defaultdict(list)
    per_param_specs = []

    for x in ns_inputs:
        if param_split_fn is None:
            split_tensors = [x]
        else:
            split_tensors = _validate_param_split_fn(param_split_fn, x, param_split_fn(x))

        param_specs = []
        for split in split_tensors:
            matrices, spec = _split_tensor_for_orthogonalization(split)
            refs = []
            for matrix in matrices:
                shape = tuple(matrix.shape)
                refs.append((shape, len(matrices_by_shape[shape])))
                matrices_by_shape[shape].append(matrix)
            param_specs.append((spec, refs))
        per_param_specs.append(param_specs)

    return matrices_by_shape, per_param_specs


def _reconstruct_orthogonalized_updates(
    orthogonalized_by_shape: dict,
    per_param_specs,
    param_recombine_fn: Optional[Callable],
) -> list[Tensor]:
    updates = []
    for param_specs in per_param_specs:
        split_updates = []
        for spec, refs in param_specs:
            matrices = [orthogonalized_by_shape[shape][idx] for shape, idx in refs]
            split_updates.append(_reconstruct_tensor_from_matrices(spec, matrices))

        if param_recombine_fn is None:
            updates.append(split_updates[0])
        else:
            updates.append(param_recombine_fn(split_updates))
    return updates


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------
class MuonBase(Optimizer8bit):
    _FSDP2_QUANTIZED_WORLD_SIZE_KEY = "__bnb_quantized_shard_world_size__"

    """
    Muon optimizer with optional 8-bit quantized momentum buffer.

    Muon (Momentum + Orthogonalization via Newton-Schulz) applies:
      1. EMA momentum update.
      2. Newton-Schulz orthogonalization of the Nesterov / momentum direction.
      3. RMS-norm LR adjustment (optional).
      4. Decoupled weight decay + parameter update.

    Only parameters with ndim >= 2 are supported. Place 1-D params (biases,
    LayerNorm weights) in a separate AdamW or SGD group.

    Arguments:
        params: Parameter groups.
        lr: Learning rate.
        momentum: EMA decay factor β (default 0.95).
        weight_decay: Decoupled weight decay coefficient (default 0.1).
        nesterov: Use Nesterov-style momentum input to NS (default True).
        adjust_lr: LR adjustment strategy applied after orthogonalization.
            - "rms_norm" (default): scale by 0.2 * sqrt(max(fan_out, fan_in))
            - "spectral_norm": scale by sqrt(fan_out / fan_in)
            - None: no adjustment
            - Callable: custom function (lr, shape) -> float
        orthogonalize_fn: Callable X -> X for orthogonalization. Defaults to
            GramNewtonSchulz when the gram_newton_schulz package is available
            (torch backend on any GPU; CuTeDSL kernels on sm90/sm100 with
            quack), else a pure-PyTorch 5-step standard Newton-Schulz with
            Polar Express coefficients.
        ns_chunk_size: Maximum number of same-shape matrices orthogonalized
            per batched NS call. Bounds the step's transient memory; the NS
            working set is roughly 3-4 chunk-sized bf16 buffers.
        optim_bits: 32 for fp32, 8 for 8-bit dynamic quantisation, 4 for 4-bit
            NF4/FP4 quantisation (~0.5 bytes/param for momentum).
        min_8bit_size: Minimum element count to use quantised storage; smaller
            parameters fall back to fp32 regardless of optim_bits.
        quant_type: 4-bit quantisation scheme when optim_bits=4.
            "nf4" (default) uses NormalFloat4 (better for momentum distributions);
            "fp4" uses floating-point 4-bit encoding.
            Ignored when optim_bits != 4.
        is_paged: Use paged (CPU-offload) memory for optimizer state.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        optim_bits: int = 32,
        min_8bit_size: int = 4096,
        quant_type: str = "nf4",
        is_paged: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if ns_chunk_size < 1:
            raise ValueError(f"Invalid ns_chunk_size: {ns_chunk_size}")
        if optim_bits not in (4, 8, 32):
            raise ValueError(f"optim_bits must be 4, 8, or 32; got {optim_bits}")
        if quant_type not in ("nf4", "fp4", "nvfp4"):
            raise ValueError(f"quant_type must be 'nf4', 'fp4', or 'nvfp4'; got {quant_type!r}")

        # Pack momentum into betas[0] so GlobalOptimManager override_config works.
        defaults = dict(
            lr=lr,
            betas=(momentum, 0.0),
            eps=1e-8,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
        )
        super().__init__(params, defaults, optim_bits=optim_bits, is_paged=is_paged)

        # Args object expected by get_config() / Optimizer8bit
        args: dict = {
            "optim_bits": optim_bits,
            "min_8bit_size": min_8bit_size,
            "max_unorm": 0.0,
            "skip_zeros": False,
        }
        self.args = MockArgs(args)
        self.optimizer_name = "muon"
        self._quant_type = quant_type  # used only when optim_bits == 4

        self._orthogonalize_fn = orthogonalize_fn if orthogonalize_fn is not None else _get_default_orthogonalize_fn()
        self.ns_chunk_size = ns_chunk_size
        # Fused Triton paths (dequant + momentum + requant + NS-input write in
        # one pass). Auto-enabled when triton imports successfully.
        self._use_fused_8bit = muon_momentum_8bit_fused is not None
        # 4-bit fused kernels: NF4 and NVFP4 have dedicated Triton kernels;
        # bitsandbytes FP4 falls back to the eager path.
        self._use_fused_4bit = muon_momentum_4bit_fused is not None and quant_type == "nf4"
        self._use_fused_nvfp4 = muon_momentum_nvfp4_fused is not None and quant_type == "nvfp4"

        if optim_bits == 8:
            self.fill_qmap()

        for group in self.param_groups:
            for p in group["params"]:
                # Reject unsupported sharded layouts (FSDP1, ZeRO-3) with a clear
                # error that points users to FSDP2.  Must run before the ndim check
                # so FSDP1 FlatParameters get the right message.
                _assert_supported_layout(p)

                # Gate paged state + FSDP2 sharding for the first cut (§7.7 / §11.B.6).
                if is_paged and _is_sharded(p):
                    raise NotImplementedError(
                        "[paged] is_paged=True is not supported together with FSDP2 "
                        "sharded parameters (DTensor).  Set is_paged=False when using "
                        "fully_shard."
                    )

                # Plain ndim < 2 check (for non-DTensor params).
                if not _is_sharded(p) and p.ndim < 2:
                    raise ValueError(
                        "MuonBase only supports parameters with 2 or more dimensions. "
                        "Place 1-D parameters (biases, norms) in a separate AdamW group."
                    )

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)
        optim_bits = config["optim_bits"]

        # Under FSDP2 (DTensor) the param is a sharded view; all state buffers
        # must be sized to the LOCAL shard so quantized block layout is consistent.
        # p_local is the plain local tensor used for device/shape queries.
        is_sharded = _is_sharded(p)
        if is_sharded:
            p_local = p.to_local()
            global_n = math.prod(tuple(p.shape))
        else:
            p_local = p
            global_n = p.numel()
        local_n = p_local.numel()

        state = self.state[p]
        state["step"] = 0

        # Fall back to fp32 for small parameters regardless of optim_bits.
        # Use global_n (world-size-independent) so the quantize-or-not decision
        # is consistent regardless of how many ranks are training.
        if global_n < config["min_8bit_size"] or optim_bits == 32:
            buf = self.get_state_buffer(p_local, dtype=torch.float32)
            # Wrap as a DTensor so DCP can gather / reshard this momentum buffer
            # across world sizes via the standard get_optimizer_state_dict API.
            state["state1"] = _as_state_dtensor(buf, p)
            return

        if optim_bits == 8:
            if "dynamic" not in self.name2qmap:
                self.fill_qmap()
            self.name2qmap["dynamic"] = self.name2qmap["dynamic"].to(p_local.device)

            s1 = self.get_state_buffer(p_local, dtype=torch.uint8)
            state["state1"] = _as_state_dtensor(s1, p)
            state["qmap1"] = self.name2qmap["dynamic"]  # replicated; stays plain

            blocksize = 256
            blocks = (local_n + blocksize - 1) // blocksize
            amax = torch.zeros((blocks,), dtype=torch.float32, device=p_local.device)
            state["absmax1"] = _as_state_dtensor(amax, p)

        elif optim_bits == 4:
            blocksize = 64
            if self._quant_type == "nvfp4":
                # NVFP4 bypasses the C++ kernel entirely; allocate directly.
                # Shape (ceil(n_local/2), 1) mirrors what quantize_4bit returns so
                # the reshape(-1) in _update_batch always produces contiguous bytes.
                n_paired = (local_n + 1) // 2
                n_blocks = (local_n + blocksize - 1) // blocksize
                s1 = torch.zeros(n_paired, 1, dtype=torch.uint8, device=p_local.device)
                state["state1"] = _as_state_dtensor(s1, p)
                amax = torch.zeros(n_blocks, dtype=torch.float32, device=p_local.device)
                state["absmax1"] = _as_state_dtensor(amax, p)
            else:
                # NF4 / FP4: call quantize_4bit once on zeros to get exactly the
                # right packed-shape (ceil(n_local/2), 1) without guessing it.
                _m_zero = torch.zeros(local_n, dtype=torch.float32, device=p_local.device)
                with torch.no_grad():
                    packed_init, quant_state_init = F.quantize_4bit(
                        _m_zero, blocksize=blocksize, quant_type=self._quant_type
                    )
                state["state1"] = _as_state_dtensor(packed_init, p)  # (ceil(n_local/2), 1)
                state["absmax1"] = _as_state_dtensor(quant_state_init.absmax, p)
                del _m_zero, packed_init, quant_state_init
            state["quant_type1"] = self._quant_type
            state["blocksize1"] = blocksize

        else:
            raise NotImplementedError(f"Unsupported optim_bits: {optim_bits}")

    # ------------------------------------------------------------------
    # Step override: batch params of same shape, run NS once per batch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not self.initialized:
            self.check_overrides()
            self.to_gpu()
            self.initialized = True

        for gindex, group in enumerate(self.param_groups):
            # Collect params with gradients; initialise state lazily
            active = []
            for pindex, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                # Re-check layout guard at step-time (params might have been
                # wrapped with FSDP2 after the optimizer was constructed).
                _assert_supported_layout(p)
                if not _is_sharded(p) and p.ndim < 2:
                    raise ValueError(
                        "MuonBase only supports parameters with 2 or more dimensions. "
                        "Place 1-D parameters (biases, norms) in a separate AdamW group."
                    )
                state = self.state[p]
                if len(state) == 0:
                    self.init_state(group, p, gindex, pindex)
                self.prefetch_state(p)
                active.append((pindex, p))

            if not active:
                continue

            # Resolve LR adjustment function once per group
            adjust_lr_fn = _resolve_adjust_lr(group["adjust_lr"])

            # Batch same-shape params together for a single NS call
            param_list = [p for _, p in active]
            for batch in _create_param_batches(param_list):
                self._update_batch(group, gindex, batch, adjust_lr_fn)

        return loss

    # ------------------------------------------------------------------
    # Batch update: momentum → quantize → NS → param update
    # ------------------------------------------------------------------
    def _update_batch(
        self,
        group: dict,
        gindex: int,
        params: list[Tensor],
        adjust_lr_fn: Callable,
    ):
        """Apply one Muon step to a batch of same-shape parameters.

        Same-shape params are processed in chunks of `ns_chunk_size`.

        For plain (non-sharded) params the existing replicated path is used:
          Phase 1 — elementwise momentum EMA → Nesterov input u_buf (per-param).
          Phase 2 — stack u_bufs → batched NS → weight-decay + param update.

        For FSDP2 sharded (DTensor) params, Pattern B (parameter-parallel) is
        used (§6 / §9 of the design note):
          Phase 1 — identical elementwise momentum EMA on the LOCAL SHARD only.
                    All quantised-state machinery stays shard-local.
          Phase 2 — gather u_buf shard to the assigned owner rank;
                    owner runs NS on the full matrix;
                    owner broadcasts the scaled update to all ranks;
                    every rank reshards the update and applies it locally.

        Owner assignment uses size-aware LPT (Longest Processing Time) greedy
        scheduling (§9.5) so NS FLOPs and peak memory are distributed across
        world_size ranks rather than duplicated.
        """
        beta = group["betas"][0]  # momentum
        nesterov = group["nesterov"]
        lr = group["lr"]
        wd = group["weight_decay"]
        param_split_fn = group.get("param_split_fn", None)
        param_recombine_fn = group.get("param_recombine_fn", None)
        if (param_split_fn is None) != (param_recombine_fn is None):
            raise ValueError("param_split_fn and param_recombine_fn must both be provided or both be None")

        shape = params[0].shape  # global shape (DTensor) or regular shape (plain)

        for start in range(0, len(params), self.ns_chunk_size):
            chunk = params[start : start + self.ns_chunk_size]
            ns_inputs: list[Tensor] = []
            any_sharded = any(_is_sharded(p) for p in chunk)

            # ----------------------------------------------------------
            # Phase 1: elementwise momentum EMA (shard-local, unchanged)
            # ----------------------------------------------------------
            for p in chunk:
                # Resolve local tensor references for FSDP2 sharded params.
                # All quantised-state buffers (state1, absmax1, qmap1) are
                # already sized to the LOCAL shard (set up in init_state).
                if _is_sharded(p):
                    p_local = p.to_local()
                    # p.grad may be a sharded or replicated DTensor.
                    grad = _local_tensor(p.grad).contiguous()
                    local_shape = p_local.shape
                    p_numel = p_local.numel()
                    p_device = p_local.device
                else:
                    p_local = p
                    # Use a contiguous view for fused Triton paths; if already
                    # contiguous this is a no-op (returns self, no allocation).
                    # p.grad is never mutated — avoids breaking view aliases,
                    # weight-tied parameters, and external grad references.
                    grad = p.grad.contiguous()
                    local_shape = shape
                    p_numel = p.numel()
                    p_device = p.device

                state = self.state[p]
                state["step"] += 1
                # u_buf is shard-sized for DTensor params, full-sized otherwise.
                u_buf = torch.empty(local_shape, dtype=torch.bfloat16, device=p_device)

                # Resolve plain local-tensor views of (possibly DTensor) state
                # buffers.  In-place writes on s1/amax propagate back through to
                # the DTensor's underlying storage so DCP always sees live data.
                s1 = _state_local(state["state1"])
                amax = _state_local(state["absmax1"]) if "absmax1" in state else None
                qmap = state.get("qmap1")  # replicated plain tensor; already local

                can_fuse_nvfp4 = (
                    "quant_type1" in state
                    and state["quant_type1"] == "nvfp4"
                    and self._use_fused_nvfp4
                    and grad.is_cuda
                    and grad.is_contiguous()
                    and s1.is_contiguous()
                    and amax.is_contiguous()
                    and u_buf.is_contiguous()
                )
                if can_fuse_nvfp4:
                    # Fused Triton NVFP4 path: single pass over data.
                    # s1 aliases the DTensor's local storage; writes go through.
                    muon_momentum_nvfp4_fused(
                        grad,
                        s1.reshape(-1),
                        amax,
                        u_buf,
                        beta,
                        nesterov,
                        blocksize=state["blocksize1"],
                    )
                    ns_inputs.append(u_buf)
                    continue

                can_fuse_4bit = (
                    "quant_type1" in state
                    and state["quant_type1"] == "nf4"
                    and self._use_fused_4bit
                    and grad.is_cuda
                    and grad.is_contiguous()
                    and s1.is_contiguous()
                    and amax.is_contiguous()
                    and u_buf.is_contiguous()
                )
                if can_fuse_4bit:
                    # Fused Triton NF4 path: single pass over data.
                    muon_momentum_4bit_fused(
                        grad,
                        s1.reshape(-1),
                        amax,
                        u_buf,
                        beta,
                        nesterov,
                        blocksize=state["blocksize1"],
                    )
                    ns_inputs.append(u_buf)
                    continue

                can_fuse_8bit = (
                    s1.dtype == torch.uint8
                    and qmap is not None
                    and self._use_fused_8bit
                    and grad.is_cuda
                    and grad.is_contiguous()
                    and s1.is_contiguous()
                    and amax.is_contiguous()
                    and qmap.is_contiguous()
                    and u_buf.is_contiguous()
                )
                if can_fuse_8bit:
                    # Fused Triton 8-bit path: one pass does dequant + momentum
                    # + requant (in place) and writes the NS input to u_buf.
                    muon_momentum_8bit_fused(
                        grad,
                        s1,
                        amax,
                        qmap,
                        u_buf,
                        beta,
                        nesterov,
                    )
                    ns_inputs.append(u_buf)
                    continue

                if s1.dtype == torch.float32:
                    m = s1
                    # m = beta*m + g (mixed-dtype add casts grad in-kernel)
                    m.mul_(beta).add_(grad)
                    u = (grad + beta * m) if nesterov else m
                elif "quant_type1" in state:
                    if state["quant_type1"] == "nvfp4":
                        # NVFP4 eager path: dequantize -> update -> requantize.
                        # Use shard-local numel/device (p_numel, p_device).
                        m = _nvfp4_dequantize_eager(
                            s1.reshape(-1),
                            amax,
                            p_numel,
                            state["blocksize1"],
                            p_device,
                        ).view(local_shape)
                        m.mul_(beta).add_(grad)
                        u = (grad + beta * m) if nesterov else m
                        _nvfp4_quantize_eager(
                            m.to(torch.float32).reshape(-1),
                            s1.reshape(-1),
                            amax,
                            state["blocksize1"],
                        )
                    else:
                        # 4-bit eager path: dequantize -> update -> requantize.
                        # QuantState.shape must be the LOCAL shard shape for FSDP2.
                        quant_state = F.QuantState(
                            absmax=amax,
                            shape=p_local.shape,
                            dtype=torch.float32,
                            blocksize=state["blocksize1"],
                            quant_type=state["quant_type1"],
                        )
                        m = F.dequantize_4bit(s1, quant_state=quant_state)
                        m = m.view(local_shape)
                        m.mul_(beta).add_(grad)
                        u = (grad + beta * m) if nesterov else m
                        # Requantize; out= and absmax= write into pre-allocated buffers.
                        F.quantize_4bit(
                            m.to(torch.float32).reshape(-1),
                            out=s1,
                            absmax=amax,
                            blocksize=state["blocksize1"],
                            quant_type=state["quant_type1"],
                        )
                else:
                    # 8-bit eager path: dequantize -> update -> requantize.
                    # dequantize_blockwise returns fp32 when given raw absmax.
                    m = F.dequantize_blockwise(
                        s1,
                        absmax=amax,
                        code=qmap,
                        blocksize=256,
                    )
                    m.mul_(beta).add_(grad)
                    u = (grad + beta * m) if nesterov else m
                    # Requantize momentum; absmax=/out= write state in place.
                    F.quantize_blockwise(
                        m,
                        code=qmap,
                        absmax=amax,
                        out=s1,
                        blocksize=256,
                    )

                u_buf.copy_(u)
                ns_inputs.append(u_buf)
                del u, m

            # ----------------------------------------------------------
            # Phase 2: NS orthogonalization + parameter update
            # ----------------------------------------------------------

            if any_sharded:
                # FSDP2 Pattern B (§6): per-param, gather-to-owner → NS → broadcast → reshard
                #
                # ns_inputs[i] is the LOCAL shard of the Nesterov direction u for
                # chunk[i].  We gather it to the assigned owner, run NS once there,
                # broadcast the scaled update back, and let every rank reshard it to
                # its local slice.  NS FLOPs and peak transient are divided across
                # world_size; momentum storage stays shard-local throughout.

                # Resolve distributed context from the first sharded param.
                sharded_p = next(p for p in chunk if _is_sharded(p))
                pg = _get_param_pg(sharded_p)
                rank = dist.get_rank(pg)
                world_size = dist.get_world_size(pg)

                # Assign owners via LPT (size-aware greedy, §9.5).
                global_shapes = [_global_matrix_shape(p) for p in chunk]
                ns_costs = [r * c * min(r, c) for (r, c) in global_shapes]
                owners = _assign_owners(ns_costs, world_size)

                for p, u_local, owner, global_shape in zip(chunk, ns_inputs, owners, global_shapes):
                    p_local = p.to_local() if _is_sharded(p) else p

                    # Gather the local Nesterov input to the owner rank.
                    # Returns the full (global) tensor on owner, None on others.
                    u_full = _gather_to_owner(u_local, p, owner)

                    if rank == owner:
                        # Run NS on the full matrix, composing param_split_fn
                        # AFTER the gather (§10.4 design note).
                        u_full_list = [u_full]
                        mats_by_shape, p_specs = _prepare_orthogonalization_inputs(u_full_list, param_split_fn)
                        orth_by_shape = {}
                        for mat_shape, mats in mats_by_shape.items():
                            batched = torch.stack(mats, dim=0)
                            orthed = self._orthogonalize_fn(batched)
                            orth_by_shape[mat_shape] = orthed.float().mul(adjust_lr_fn(lr, mat_shape))
                        updates_full = _reconstruct_orthogonalized_updates(orth_by_shape, p_specs, param_recombine_fn)
                        # Reshape update to match the param's global shape.
                        upd_full = updates_full[0].reshape(p.shape).to(torch.float32)
                    else:
                        upd_full = None

                    # Broadcast update from owner to all ranks, then reshard to
                    # this rank's local slice (uses DTensor redistribute).
                    upd_local = _broadcast_and_reshard(upd_full, p, owner, p_local.shape)

                    # Decoupled weight decay + parameter update on the LOCAL shard.
                    p_local.mul_(1.0 - lr * wd).add_(upd_local.to(dtype=p_local.dtype), alpha=-1.0)

                del ns_inputs
                continue

            # Non-sharded fast path: 2-D params with no param_split_fn.
            # Preserved exactly so CPU 32-bit tests compare with tight tolerances.
            if param_split_fn is None and len(shape) == 2:
                stacked = torch.stack(ns_inputs, dim=0)
                orthogonalized = self._orthogonalize_fn(stacked)
                adjusted_lr = adjust_lr_fn(lr, shape)
                for p, update in zip(chunk, orthogonalized.unbind(0)):
                    p.data.mul_(1.0 - lr * wd).add_(update, alpha=-adjusted_lr)
                del stacked, orthogonalized
                continue

            # Non-sharded general path (3-D / n-D / param_split_fn).
            matrices_by_shape, per_param_specs = _prepare_orthogonalization_inputs(ns_inputs, param_split_fn)
            orthogonalized_by_shape = {}
            for matrix_shape, matrices in matrices_by_shape.items():
                batched_input = torch.stack(matrices, dim=0)
                orthogonalized = self._orthogonalize_fn(batched_input)
                orthogonalized_by_shape[matrix_shape] = orthogonalized.float().mul(adjust_lr_fn(lr, matrix_shape))

            updates = _reconstruct_orthogonalized_updates(
                orthogonalized_by_shape,
                per_param_specs,
                param_recombine_fn,
            )

            # --- Decoupled weight decay + parameter update ---
            for p, update in zip(chunk, updates):
                if update.shape != p.shape:
                    raise RuntimeError(
                        f"Muon orthogonalized update shape {tuple(update.shape)} does not match "
                        f"parameter shape {tuple(p.shape)}"
                    )
                p.data.mul_(1.0 - lr * wd).add_(update.to(dtype=p.dtype), alpha=-1.0)

            del ns_inputs, matrices_by_shape, orthogonalized_by_shape, updates

    # ------------------------------------------------------------------
    # state_dict / load_state_dict: FSDP2 checkpoint override
    # ------------------------------------------------------------------

    def state_dict(self):
        """Return optimizer state dict compatible with PyTorch DCP.

        For FSDP2 (DTensor) params, state1 and absmax1 are already live
        DTensors in ``optimizer.state``.  We bypass Optimizer8bit's
        ``non_castable_tensor_keys`` wrapping so that DCP's
        ``get_optimizer_state_dict`` and ``dcp.save`` can see and reshard them
        directly.  Quantized state is tagged with the current world size so
        ``load_state_dict`` can reject unsupported cross-world-size reshards.

        For non-sharded params the Optimizer8bit behaviour is preserved
        (non_castable tensors are wrapped behind ``_FSDP_WRAPPED_QUANT_STATE_KEY``
        for FSDP1 / plain ``torch.save`` compatibility).
        """
        all_params = list(chain.from_iterable(g["params"] for g in self.param_groups))
        any_sharded = any(_is_sharded(p) for p in all_params)

        if not any_sharded:
            return super().state_dict()

        # Use the PyTorch base-class state_dict: state1/absmax1 are already
        # DTensors, so they appear correctly without any extra wrapping.
        raw = torch.optim.Optimizer.state_dict(self)

        # Shallow-copy each per-param dict to avoid mutating live optimizer state
        # when we add the world-size tag below.
        raw["state"] = {
            k: {kk: vv for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in raw["state"].items()
        }

        for idx, param_state in raw["state"].items():
            if not isinstance(param_state, dict):
                continue
            p = all_params[idx]
            if not _is_sharded(p):
                continue
            # Tag quantized (8/4-bit) sharded state with the saving world size
            # so load_state_dict can raise a clear error when cross-world-size
            # reshard is attempted (fp32 reshards cleanly; quantized does not).
            if "qmap1" in param_state or "quant_type1" in param_state:
                param_state[self._FSDP2_QUANTIZED_WORLD_SIZE_KEY] = _param_shard_world_size(p)

        return raw

    def load_state_dict(self, state_dict, move_to_device=True):
        """Load optimizer state for FSDP2 params after DCP has resharded it.

        Call pattern (worker)::

            template = muon_opt.state_dict()         # DTensors with current mesh
            dcp.load({"opt": template}, ...)         # DCP fills/reshards in-place
            muon_opt.load_state_dict(template)       # adopt the loaded state

        The DTensors for state1/absmax1 in *template* have already been
        resharded by ``dcp.load`` to the current world size; this method
        validates quantized-reshard safety and then adopts them directly
        into the live optimizer state (no extra redistribute call needed).

        For non-sharded params the Optimizer8bit logic is used unchanged.
        """
        from copy import deepcopy

        all_params = list(chain.from_iterable(g["params"] for g in self.param_groups))
        any_sharded = any(_is_sharded(p) for p in all_params)

        if not any_sharded:
            return super().load_state_dict(state_dict, move_to_device=move_to_device)

        # Deep-copy to avoid mutating the caller's dict.
        state_dict = deepcopy(state_dict)

        groups = self.param_groups
        saved_groups = state_dict["param_groups"]
        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
            )

        # Map saved integer param index → current param object.
        id_map = {
            old_id: p
            for old_id, p in zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        }

        new_state: dict = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k not in id_map:
                new_state[k] = v
                continue
            p = id_map[k]
            if not isinstance(v, dict):
                new_state[p] = v
                continue

            # Validate cross-world-size quantized reshard: 8/4-bit block
            # boundaries are shard-local, so resharding changes absmax layout.
            saved_ws = v.get(self._FSDP2_QUANTIZED_WORLD_SIZE_KEY)
            if saved_ws is not None and _is_sharded(p):
                current_ws = _param_shard_world_size(p)
                if int(saved_ws) != current_ws:
                    raise NotImplementedError(
                        "Muon 8/4-bit FSDP2 optimizer checkpoint reshard across "
                        "world sizes is not supported: quantized momentum uses "
                        "shard-local blockwise absmax, so block boundaries change "
                        "when the shard size changes. Use Muon32bit for exact "
                        "cross-world-size optimizer checkpointing."
                    )

            param_state: dict = {}
            for sk, sv in v.items():
                if sk == self._FSDP2_QUANTIZED_WORLD_SIZE_KEY:
                    continue  # metadata only; not a real state entry
                if isinstance(sv, Tensor):
                    # DTensors (state1, absmax1): dcp.load has already resharded
                    # them to the current world size — adopt directly as live state.
                    # Plain tensors (qmap1, etc.): optionally move to device.
                    if move_to_device and not (DTensor is not None and isinstance(sv, DTensor)):
                        target_device = p.to_local().device if _is_sharded(p) else p.device
                        sv = sv.to(target_device)
                    param_state[sk] = sv
                else:
                    param_state[sk] = sv
            new_state[p] = param_state

        def update_group(group, new_group):
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": new_state, "param_groups": param_groups})

    # Satisfy abstract interface (not used since we override step())
    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        pass  # Logic lives in _update_batch; this method is never called.


# ---------------------------------------------------------------------------
# Public API: Muon, Muon8bit, Muon32bit
# ---------------------------------------------------------------------------
class Muon(MuonBase):
    """
    Muon optimizer (32-bit momentum buffer by default).

    For quantized variants see :class:`Muon8bit` and :class:`Muon4bit`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        optim_bits: int = 32,
        min_8bit_size: int = 4096,
        quant_type: str = "nf4",
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=optim_bits,
            min_8bit_size=min_8bit_size,
            quant_type=quant_type,
            is_paged=is_paged,
        )


class Muon8bit(MuonBase):
    """
    Muon optimizer with 8-bit quantized momentum buffer.

    Persistent state is ~1 byte/param (uint8 + absmax overhead), giving
    roughly half the footprint of AdamW8bit for the same model.

    See :class:`MuonBase` for full documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=8,
            min_8bit_size=min_8bit_size,
            is_paged=is_paged,
        )


class Muon32bit(MuonBase):
    """
    Muon optimizer with 32-bit (fp32) momentum buffer.

    See :class:`MuonBase` for full documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=32,
            min_8bit_size=min_8bit_size,
            is_paged=is_paged,
        )


class Muon4bit(MuonBase):
    """
    Muon optimizer with 4-bit quantized momentum buffer.

    Persistent state is ~0.5 bytes/param (two NF4/FP4 codes packed per byte,
    blocksize=64) plus a small fp32 absmax vector (~1/64 bytes/param overhead),
    giving a total of ~0.52 bytes/param — roughly half the footprint of
    Muon8bit and ~8x less than AdamW.

    NF4 (NormalFloat4) is recommended for momentum buffers because the
    quantisation levels are optimal for normally-distributed data.  FP4 is
    provided as an alternative for data with heavy tails.

    When Triton is available and quant_type="nf4", the momentum update
    (dequantise + EMA + Nesterov + requantise + write NS input) is fused
    into a single kernel pass.  FP4 uses the eager path (quantize_4bit /
    dequantize_4bit from bitsandbytes.functional).

    See :class:`MuonBase` for full documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        quant_type: str = "nf4",
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=4,
            min_8bit_size=min_8bit_size,
            quant_type=quant_type,
            is_paged=is_paged,
        )


class Muon4bitNVFP4(MuonBase):
    """
    Muon optimizer with NVIDIA FP4 (e2m1) quantized momentum buffer.

    Momentum storage: ~0.52 bytes/param (packed e2m1 uint8, blocksize=64
    + fp32 absmax overhead) — same footprint as Muon4bit(quant_type='nf4').

    On Blackwell (sm100, B200/B300) the Newton-Schulz X @ X.T Gram GEMM is
    routed through ``torch._scaled_grouped_mm_v2`` using float4_e2m1fn_x2
    inputs with float8_e4m3fn block scales (BlockWise1x16, SWIZZLE_32_4_4),
    activating native FP4 tensor core instructions (tcgen05.mma).  All other
    NS ops (A@A, B@X update, polynomial) stay in BF16.  On non-sm100 hardware
    the standard BF16 NS path is used automatically.

    The elementwise momentum update (dequant → EMA → requant) always runs as
    a fused Triton kernel using software NVFP4 encoding — there are no scalar
    FP4 ALU instructions on sm100; only tensor cores use FP4.

    See :class:`MuonBase` for full parameter documentation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.1,
        nesterov: bool = True,
        adjust_lr: str | Callable | None = "rms_norm",
        orthogonalize_fn: Optional[Callable[[Tensor], Tensor]] = None,
        ns_chunk_size: int = 8,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        # Activate the FP4 tensor-core NS path only on data-center Blackwell
        # (sm100, B200/B300). It relies on BlockWise1x16 NVFP4 scaled GEMM
        # (torch._scaled_grouped_mm_v2) which is not available on consumer/
        # workstation Blackwell (sm120) or earlier; those use the standard
        # bf16 NS path. Capabilities mirror _make_default_orthogonalize_fn.
        if orthogonalize_fn is None:
            try:
                cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
                if cap in ((10, 0), (10, 3)):
                    orthogonalize_fn = _make_sm100_nvfp4_ns_fn()
            except Exception:
                pass
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adjust_lr=adjust_lr,
            orthogonalize_fn=orthogonalize_fn,
            ns_chunk_size=ns_chunk_size,
            optim_bits=4,
            min_8bit_size=min_8bit_size,
            quant_type="nvfp4",
            is_paged=is_paged,
        )

from bhaskera.trainer.optimizer_registry import register_optimizer

@register_optimizer("muon")
def build_muon(model, train_cfg):
    """
    Builder function called by the Bhaskera framework.
    """
    from bhaskera.trainer.optim import _get_default_param_groups
    
    opt_cfg = train_cfg.optimizer
    
    # Extract kwargs, falling back to base training config if not specified
    lr = opt_cfg.kwargs.get("lr", train_cfg.lr)
    weight_decay = opt_cfg.kwargs.get("weight_decay", train_cfg.weight_decay)
    
    momentum = opt_cfg.kwargs.get("momentum", 0.95)
    nesterov = opt_cfg.kwargs.get("nesterov", True)
    adjust_lr = opt_cfg.kwargs.get("adjust_lr", "rms_norm")
    ns_chunk_size = opt_cfg.kwargs.get("ns_chunk_size", 8)
    optim_bits = opt_cfg.kwargs.get("optim_bits", 8)
    min_8bit_size = opt_cfg.kwargs.get("min_8bit_size", 4096)
    quant_type = opt_cfg.kwargs.get("quant_type", "nf4")
    
    param_groups = _get_default_param_groups(model, weight_decay)
    
    return Muon(
        param_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        adjust_lr=adjust_lr,
        ns_chunk_size=ns_chunk_size,
        optim_bits=optim_bits,
        min_8bit_size=min_8bit_size,
        quant_type=quant_type
    )
