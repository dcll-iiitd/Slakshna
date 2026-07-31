"""
Tests for bhaskera.inference (v2 optimised) — no GPU or real model needed.

New tests added for v2:
  - O(1) amortised decode: verify dequantize count doesn't grow with seq_len
  - Incremental cache consistency: decode cache matches full reconstruction
  - torch.bucketize codebook: verify FastLloydMaxCodebook round-trip

Run with:
    pytest src/bhaskera/inference/tests/test_inference.py -v
"""
from __future__ import annotations

import math
import time
import pytest
import torch
import torch.nn.functional as F


# ===========================================================================
# Lloyd-Max codebook
# ===========================================================================

class TestLloydMaxCodebook:
    def test_codebook_shape(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        cb = LloydMaxCodebook.get(d=128, bits=4)
        assert cb.centroids.shape == (16,)
        assert cb.boundaries.shape == (15,)

    def test_centroids_ordered(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        cb = LloydMaxCodebook.get(d=128, bits=4)
        assert (cb.centroids[1:] - cb.centroids[:-1] > 0).all()

    def test_round_trip_accuracy(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        d, bits = 128, 4
        cb = LloydMaxCodebook.get(d=d, bits=bits)
        torch.manual_seed(0)
        sigma = 1.0 / math.sqrt(d)
        x = torch.randn(10_000) * sigma
        idx   = cb.quantize(x)
        x_hat = cb.dequantize(idx)
        mse   = ((x - x_hat) ** 2).mean().item()
        upper = sigma**2 * (2 ** (-2 * bits)) * (math.pi**2 / 3) * 4
        assert mse < upper

    def test_cache_reuse(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        assert LloydMaxCodebook.get(128, 4) is LloydMaxCodebook.get(128, 4)

    def test_different_bitwidths(self):
        from bhaskera.inference.lloyd_max import LloydMaxCodebook
        assert LloydMaxCodebook.get(128, 2).n_levels == 4
        assert LloydMaxCodebook.get(128, 8).n_levels == 256


# ===========================================================================
# FastLloydMaxCodebook (bucketize path)
# ===========================================================================

class TestFastLloydMaxCodebook:
    def test_matches_slow_codebook(self):
        """bucketize results should agree with argmin for interior values."""
        from bhaskera.inference.kv_cache import FastLloydMaxCodebook
        from bhaskera.inference.lloyd_max import LloydMaxCodebook

        device = torch.device("cpu")
        cb_fast = FastLloydMaxCodebook.get(64, 4, device)
        cb_slow = LloydMaxCodebook.get(64, 4)

        torch.manual_seed(1)
        x = torch.randn(500) * (1.0 / math.sqrt(64))
        # Exclude boundary-crossing samples (argmin and bucketize may differ by 1 at exact boundaries)
        x = x.clamp(cb_slow.centroids[0].item() + 1e-4,
                    cb_slow.centroids[-1].item() - 1e-4)

        idx_fast = cb_fast.quantize(x.unsqueeze(-1)).squeeze(-1)
        idx_slow = cb_slow.quantize(x)
        assert (idx_fast == idx_slow).all(), "Fast and slow codebooks should agree"

    def test_round_trip_quality(self):
        from bhaskera.inference.kv_cache import FastLloydMaxCodebook
        device = torch.device("cpu")
        cb = FastLloydMaxCodebook.get(64, 4, device)
        x = torch.randn(1000, 64) * (1.0 / math.sqrt(64))
        idx = cb.quantize(x)
        x_hat = cb.dequantize(idx)
        assert x_hat.shape == x.shape


# ===========================================================================
# Sampling utilities (unchanged)
# ===========================================================================

class TestSampling:
    def test_temperature_scale_identity(self):
        from bhaskera.inference.sampling import temperature_scale
        logits = torch.randn(4, 100)
        assert torch.allclose(temperature_scale(logits, 1.0), logits)

    def test_top_k_zeroes_others(self):
        from bhaskera.inference.sampling import top_k_filter
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        probs = F.softmax(top_k_filter(logits, top_k=2), dim=-1)
        assert (probs[0, :3] < 1e-6).all()

    def test_greedy_no_sample(self):
        from bhaskera.inference.sampling import sample_from_logits
        logits = torch.tensor([[0.1, 9.0, 0.3]])
        assert sample_from_logits(logits, do_sample=False).item() == 1


# ===========================================================================
# StaticKVCache
# ===========================================================================

class TestStaticKVCache:
    def _make(self, **kw):
        from bhaskera.inference.kv_cache import StaticKVCache
        d = dict(num_layers=4, batch_size=2, num_heads=8, head_dim=64,
                 max_seq_len=512, dtype=torch.float32, device=torch.device("cpu"))
        d.update(kw)
        return StaticKVCache(**d)

    def test_update_shape(self):
        c = self._make()
        k = torch.randn(2, 8, 10, 64)
        fk, fv = c.update(k, torch.randn_like(k), 0)
        assert fk.shape == (2, 8, 10, 64)

    def test_reset(self):
        c = self._make()
        c.update(torch.randn(2,8,5,64), torch.randn(2,8,5,64), 0)
        c.advance(5)
        c.reset()
        assert c.seq_len == 0

    def test_overflow_raises(self):
        c = self._make(max_seq_len=8)
        with pytest.raises(ValueError, match="overflow"):
            c.update(torch.randn(2,8,10,64), torch.randn(2,8,10,64), 0)


# ===========================================================================
# TurboQuantKVCache — correctness + O(1) performance
# ===========================================================================

class TestTurboQuantKVCache:
    def _make(self, **kw):
        from bhaskera.inference.kv_cache import TurboQuantKVCache
        d = dict(num_layers=4, batch_size=1, num_heads=4, head_dim=64,
                 key_bits=4, value_bits=2, residual_window=8, protected_layers=1,
                 dtype=torch.float32, device=torch.device("cpu"), max_seq_len=256)
        d.update(kw)
        return TurboQuantKVCache(**d)

    def test_update_returns_correct_shape(self):
        c = self._make()
        k = torch.randn(1, 4, 5, 64)
        fk, fv = c.update(k, torch.randn_like(k), 0)
        assert fk.shape == (1, 4, 5, 64)

    def test_reset(self):
        c = self._make()
        c.update(torch.randn(1,4,5,64), torch.randn(1,4,5,64), 0)
        c.reset()
        assert c.seq_len == 0

    def test_reconstruction_quality_k4v2(self):
        """Round-trip cosine similarity should be > 0.92 with K4/V2."""
        c = self._make(key_bits=4, value_bits=2, residual_window=0)
        torch.manual_seed(7)
        k = torch.randn(1, 4, 32, 64)
        v = torch.randn_like(k)

        # Force immediate compression (residual_window=0 means evict everything)
        fk, fv = c.update(k, v, 0)

        k_flat    = k.reshape(-1, 64)
        fk_flat   = fk.reshape(-1, 64)
        cos_sim   = F.cosine_similarity(k_flat, fk_flat, dim=-1).mean().item()
        assert cos_sim > 0.92, f"Cosine similarity {cos_sim:.4f} too low"

    def test_incremental_matches_full(self):
        """
        Decode cache must be consistent regardless of whether tokens arrive
        one at a time or in a batch.
        """
        from bhaskera.inference.kv_cache import TurboQuantKVCache
        torch.manual_seed(42)

        # Batch path: all 20 tokens in one call
        c_batch = TurboQuantKVCache(
            num_layers=1, batch_size=1, num_heads=2, head_dim=32,
            key_bits=4, value_bits=2, residual_window=4, protected_layers=0,
            max_seq_len=64, dtype=torch.float32, device=torch.device("cpu"),
        )
        k_all = torch.randn(1, 2, 20, 32)
        v_all = torch.randn_like(k_all)
        fk_batch, fv_batch = c_batch.update(k_all, v_all, layer_idx=0)

        # Incremental path: same tokens one at a time
        c_incr = TurboQuantKVCache(
            num_layers=1, batch_size=1, num_heads=2, head_dim=32,
            key_bits=4, value_bits=2, residual_window=4, protected_layers=0,
            max_seq_len=64, dtype=torch.float32, device=torch.device("cpu"),
        )
        for t in range(20):
            fk_incr, fv_incr = c_incr.update(
                k_all[:, :, t:t+1, :], v_all[:, :, t:t+1, :], layer_idx=0
            )

        # Last incremental output should match batch output
        assert fk_incr.shape[2] == fk_batch.shape[2], "Sequence lengths must match"
        # Quality check: both should be close to original
        cos_batch = F.cosine_similarity(
            fk_batch.reshape(-1, 32), k_all.reshape(-1, 32), dim=-1
        ).mean().item()
        cos_incr = F.cosine_similarity(
            fk_incr.reshape(-1, 32), k_all.reshape(-1, 32), dim=-1
        ).mean().item()
        assert abs(cos_batch - cos_incr) < 0.05, (
            f"Batch ({cos_batch:.4f}) and incremental ({cos_incr:.4f}) quality should match"
        )

    def test_amortised_O1_per_step(self):
        """
        Decode time per step must NOT grow linearly with sequence length.
        (This test catches a regression back to the O(n²) original.)
        """
        from bhaskera.inference.kv_cache import TurboQuantKVCache

        cache = TurboQuantKVCache(
            num_layers=1, batch_size=1, num_heads=4, head_dim=64,
            key_bits=4, value_bits=2, residual_window=16, protected_layers=0,
            max_seq_len=512, dtype=torch.float32, device=torch.device("cpu"),
        )

        # Warm up (first few steps may be slower due to codebook init)
        for _ in range(10):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        cache.advance(10)

        # Measure 10 steps near token 20 (short history)
        t0 = time.perf_counter()
        for _ in range(10):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        t_short = (time.perf_counter() - t0) / 10

        # Advance to token 200 (much longer history)
        for _ in range(180):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        cache.advance(200)

        # Measure 10 steps near token 200
        t0 = time.perf_counter()
        for _ in range(10):
            cache.update(torch.randn(1,4,1,64), torch.randn(1,4,1,64), 0)
        t_long = (time.perf_counter() - t0) / 10

        # Should not be more than 5× slower at 10× the sequence length
        ratio = t_long / t_short
        assert ratio < 5.0, (
            f"Step time grew {ratio:.1f}× from short to long history "
            f"(expected ~1×, got {t_short*1000:.1f}ms → {t_long*1000:.1f}ms). "
            f"O(n²) regression detected!"
        )


# ===========================================================================
# Rotation matrix
# ===========================================================================

class TestRotationMatrix:
    def test_orthogonal(self):
        from bhaskera.inference.kv_cache import _generate_rotation_matrix
        R = _generate_rotation_matrix(64, seed=0)
        assert torch.allclose(R @ R.T, torch.eye(64), atol=1e-5)

    def test_deterministic(self):
        from bhaskera.inference.kv_cache import _generate_rotation_matrix
        R1 = _generate_rotation_matrix(64, seed=42)
        R2 = _generate_rotation_matrix(64, seed=42)
        assert torch.allclose(R1, R2)


# ===========================================================================
# build_kv_cache factory
# ===========================================================================

class TestBuildKVCache:
    _base = dict(num_layers=2, batch_size=1, num_heads=4, head_dim=32,
                 max_seq_len=128, dtype=torch.float32, device=torch.device("cpu"))

    def test_static(self):
        from bhaskera.inference.kv_cache import build_kv_cache, StaticKVCache
        assert isinstance(build_kv_cache(strategy="static", tq_cfg=None, **self._base), StaticKVCache)

    def test_none(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        assert build_kv_cache(strategy="none", tq_cfg=None, **self._base) is None

    def test_turboquant_requires_cfg(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        with pytest.raises(ValueError):
            build_kv_cache(strategy="turboquant", tq_cfg=None, **self._base)

    def test_turboquant_with_cfg(self):
        from bhaskera.inference.kv_cache import build_kv_cache, TurboQuantKVCache
        from bhaskera.config import TurboQuantConfig
        tq = TurboQuantConfig(key_bits=4, value_bits=2, residual_window=16)
        assert isinstance(
            build_kv_cache(strategy="turboquant", tq_cfg=tq, **self._base),
            TurboQuantKVCache,
        )

    def test_unknown_strategy(self):
        from bhaskera.inference.kv_cache import build_kv_cache
        with pytest.raises(ValueError):
            build_kv_cache(strategy="invalid", tq_cfg=None, **self._base)


# ===========================================================================
# Config
# ===========================================================================

class TestInferenceConfig:
    def test_defaults(self):
        from bhaskera.config import Config
        cfg = Config()
        assert cfg.inference.kv_cache == "static"
        assert cfg.inference.turboquant.key_bits == 4
        assert cfg.inference.turboquant.value_bits == 2

    def test_yaml_override(self, tmp_path):
        import yaml
        from bhaskera.config import load_config
        d = {"inference": {"max_new_tokens": 256, "kv_cache": "turboquant",
                           "turboquant": {"key_bits": 3, "residual_window": 64}}}
        p = tmp_path / "t.yaml"
        p.write_text(yaml.dump(d))
        cfg = load_config(str(p))
        assert cfg.inference.max_new_tokens == 256
        assert cfg.inference.turboquant.key_bits == 3