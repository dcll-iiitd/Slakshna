"""
bhaskera.inference.speculative
================================
Speculative decoding for 2–3× decode speedup with zero quality loss.

Algorithm (Leviathan et al. 2023 — arXiv:2211.17192):
  1. A small *draft* model generates `num_draft_tokens` tokens autoregressively.
  2. The *target* model evaluates all draft tokens in a single parallel forward
     pass (much cheaper than n × sequential target passes).
  3. Each draft token is accepted or rejected via rejection sampling:
       - If p_target(x) >= p_draft(x): accept deterministically.
       - Otherwise: accept with probability p_target(x) / p_draft(x).
  4. The first rejected token is resampled from a corrected distribution.
     All tokens after the first rejection are discarded.

This is *lossless*: the output distribution is identical to greedy/sampling
from the target model alone, just with fewer target forward passes.

Requirements:
  - Draft and target must share the same vocabulary (tokenizer).
  - Draft should be ≥ 5–10× smaller for meaningful speedup (e.g. 70M vs 7B).
  - Both models must be on the same device.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpeculativeDecoder:
    """Speculative decoding wrapper around a target + draft model pair.

    Args:
        target_model:     The main (large) language model.
        draft_model:      The small speculative model. If None, falls back to
                          standard autoregressive decoding from the target.
        num_draft_tokens: Number of tokens the draft generates per step.
        temperature:      Sampling temperature applied to *both* models.
        top_p:            Nucleus threshold applied to *both* models.
        top_k:            Top-k applied to *both* models.
        device:           Torch device.
    """

    def __init__(
        self,
        target_model: torch.nn.Module,
        draft_model: Optional[torch.nn.Module],
        num_draft_tokens: int = 5,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        self.target_model     = target_model
        self.draft_model      = draft_model
        self.num_draft_tokens = num_draft_tokens
        self.temperature      = temperature
        self.top_p            = top_p
        self.top_k            = top_k
        self.device           = device
        self._has_draft       = draft_model is not None

        if not self._has_draft:
            logger.warning(
                "SpeculativeDecoder: no draft model provided — "
                "will use standard autoregressive decoding from target."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_step(
        self,
        input_ids: torch.Tensor,
        target_past_kv=None,
        draft_past_kv=None,
    ) -> Tuple[torch.Tensor, object, object]:
        """Generate one accepted token (or up to num_draft_tokens if all accepted).

        Args:
            input_ids:       (batch, seq_len) current token sequence.
            target_past_kv:  Past KV state for the target model.
            draft_past_kv:   Past KV state for the draft model.

        Returns:
            new_tokens:      (batch, n_accepted) accepted token ids.
            target_past_kv:  Updated target model KV cache.
            draft_past_kv:   Updated draft model KV cache.
        """
        if not self._has_draft:
            return self._standard_step(input_ids, target_past_kv)

        return self._speculative_step(input_ids, target_past_kv, draft_past_kv)

    # ------------------------------------------------------------------
    # Standard fallback (no draft)
    # ------------------------------------------------------------------

    def _standard_step(
        self,
        input_ids: torch.Tensor,
        past_kv=None,
    ) -> Tuple[torch.Tensor, object, None]:
        out = self.target_model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]          # (batch, vocab)
        next_token = self._sample(logits)       # (batch,)
        return next_token.unsqueeze(1), out.past_key_values, None

    # ------------------------------------------------------------------
    # Core speculative decoding step
    # ------------------------------------------------------------------

    def _speculative_step(
        self,
        input_ids: torch.Tensor,
        target_past_kv=None,
        draft_past_kv=None,
    ) -> Tuple[torch.Tensor, object, object]:
        batch = input_ids.shape[0]

        # ── Phase 1: Draft generates num_draft_tokens ──────────────────
        draft_tokens, draft_probs, draft_past_kv = self._draft_generate(
            input_ids, draft_past_kv
        )
        # draft_tokens: (batch, num_draft_tokens)
        # draft_probs:  (batch, num_draft_tokens, vocab)

        # ── Phase 2: Target evaluates all draft tokens in ONE pass ─────
        # Concatenate input with draft tokens for a single forward
        full_input = torch.cat([input_ids, draft_tokens], dim=1)  # (B, S+K)

        target_out = self.target_model(
            input_ids=full_input,
            past_key_values=target_past_kv,
            use_cache=True,
        )
        # Logits for positions corresponding to draft tokens + one extra
        # target_logits: (batch, num_draft_tokens + 1, vocab)
        # We want the last (num_draft_tokens+1) positions
        target_logits = target_out.logits[:, -self.num_draft_tokens - 1:, :]
        target_past_kv = target_out.past_key_values

        target_probs = self._logits_to_probs(target_logits)
        # target_probs: (batch, num_draft_tokens + 1, vocab)

        # ── Phase 3: Rejection sampling ────────────────────────────────
        accepted_tokens = self._rejection_sample(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs[:, :-1, :],    # positions 0..K-1
            bonus_logits=target_logits[:, -1, :],     # position K (free bonus)
        )
        # accepted_tokens: (batch, n_accepted) where 1 <= n_accepted <= K+1

        return accepted_tokens, target_past_kv, draft_past_kv

    # ------------------------------------------------------------------
    # Draft generation
    # ------------------------------------------------------------------

    def _draft_generate(
        self,
        input_ids: torch.Tensor,
        past_kv=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, object]:
        """Run the draft model for num_draft_tokens steps.

        Returns:
            tokens: (batch, num_draft_tokens) generated token ids.
            probs:  (batch, num_draft_tokens, vocab) softmax probabilities.
            past_kv: updated draft KV cache.
        """
        tokens_list: List[torch.Tensor] = []
        probs_list:  List[torch.Tensor] = []
        cur_input = input_ids

        for _ in range(self.num_draft_tokens):
            out = self.draft_model(
                input_ids=cur_input,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]         # (batch, vocab)
            probs  = self._logits_to_probs(logits.unsqueeze(1)).squeeze(1)
            tok    = self._sample(logits)          # (batch,)
            past_kv = out.past_key_values

            tokens_list.append(tok)
            probs_list.append(probs)
            cur_input = tok.unsqueeze(1)           # next input = just this token

        tokens = torch.stack(tokens_list, dim=1)   # (batch, K)
        probs  = torch.stack(probs_list, dim=1)    # (batch, K, vocab)
        return tokens, probs, past_kv

    # ------------------------------------------------------------------
    # Rejection sampling
    # ------------------------------------------------------------------

    def _rejection_sample(
        self,
        draft_tokens: torch.Tensor,   # (batch, K)
        draft_probs:  torch.Tensor,   # (batch, K, vocab)
        target_probs: torch.Tensor,   # (batch, K, vocab)
        bonus_logits: torch.Tensor,   # (batch, vocab) — free token from target
    ) -> torch.Tensor:
        """Standard speculative decoding rejection sampling.

        For token i:
          - Acceptance prob = min(1, p_target(x_i) / p_draft(x_i))
          - If rejected: resample from normalised(max(0, p_target - p_draft))
          - All tokens after first rejection are discarded.
          - Bonus token from target appended if all drafts accepted.
        """
        batch, K = draft_tokens.shape
        device = draft_tokens.device
        accepted_list = []

        # Process each position in the draft
        for i in range(K):
            tok_i   = draft_tokens[:, i]                      # (batch,)
            p_draft = draft_probs[:, i, :].gather(
                1, tok_i.unsqueeze(1)
            ).squeeze(1).clamp(min=1e-9)                       # (batch,)
            p_target = target_probs[:, i, :].gather(
                1, tok_i.unsqueeze(1)
            ).squeeze(1).clamp(min=0.0)                        # (batch,)

            # Acceptance probability
            accept_prob = (p_target / p_draft).clamp(max=1.0)  # (batch,)
            u = torch.rand(batch, device=device)
            accept = u < accept_prob                            # (batch,) bool

            if accept.all():
                # All batch items accept this token
                accepted_list.append(tok_i)
            else:
                # Some items reject — resample for rejected items
                # Corrected distribution: max(0, p_target - p_draft)
                corrected = (target_probs[:, i, :] - draft_probs[:, i, :]).clamp(min=0.0)
                corrected_sum = corrected.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                corrected_probs = corrected / corrected_sum

                resampled = torch.multinomial(corrected_probs, 1).squeeze(1)  # (batch,)
                # For accepted items keep draft token; for rejected use resample
                tok_final = torch.where(accept, tok_i, resampled)
                accepted_list.append(tok_final)
                # Stop here — all tokens after first rejection are invalid
                break
        else:
            # All K drafts accepted — append free bonus token from target
            bonus_tok = self._sample(bonus_logits)
            accepted_list.append(bonus_tok)

        return torch.stack(accepted_list, dim=1)  # (batch, n_accepted)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities with temperature/top-k/top-p."""
        from .sampling import temperature_scale, top_k_filter, top_p_filter
        l = logits.clone()
        if self.temperature != 1.0:
            l = temperature_scale(l, self.temperature)
        if self.top_k > 0:
            l = top_k_filter(l, self.top_k)
        if self.top_p < 1.0:
            l = top_p_filter(l, self.top_p)
        return F.softmax(l, dim=-1)

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from logits using configured temperature/top-k/top-p."""
        from .sampling import sample_from_logits
        return sample_from_logits(
            logits,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=True,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_speculative_decoder(
    target_model: torch.nn.Module,
    cfg,               # SpeculativeConfig
    infer_cfg,         # InferenceConfig
    device: torch.device,
) -> Optional[SpeculativeDecoder]:
    """Build a SpeculativeDecoder if config.speculative.enabled is True.

    Loads the draft model from HuggingFace using the same dtype as target.
    Returns None if speculative decoding is disabled.
    """
    if not cfg.enabled:
        return None

    if not cfg.draft_model_name:
        logger.warning(
            "Speculative decoding enabled but no draft_model_name specified — disabling."
        )
        return None

    logger.info(f"Loading draft model: {cfg.draft_model_name}")
    try:
        from transformers import AutoModelForCausalLM

        target_dtype = next(target_model.parameters()).dtype
        draft_model = AutoModelForCausalLM.from_pretrained(
            cfg.draft_model_name,
            torch_dtype=target_dtype,
            device_map=str(device),
        )
        draft_model.eval()
        logger.info(f"Draft model loaded: {cfg.draft_model_name}")
    except Exception as e:
        logger.error(f"Failed to load draft model '{cfg.draft_model_name}': {e}")
        logger.warning("Falling back to standard autoregressive decoding.")
        draft_model = None

    return SpeculativeDecoder(
        target_model=target_model,
        draft_model=draft_model,
        num_draft_tokens=cfg.num_draft_tokens,
        temperature=infer_cfg.temperature,
        top_p=infer_cfg.top_p,
        top_k=infer_cfg.top_k,
        device=device,
    )
