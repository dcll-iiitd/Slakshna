"""
bhaskera.inference.param2
==========================
Param2-17B-A2.4B-Thinking inference support.

Architecture facts (from HF model card + config):
  - Hybrid MoE: 17B total / 2.4B active per token
  - 21 hidden layers, hidden_size=2048, intermediate_size=7168
  - 64 routed experts, top-6 routing, 2 shared experts (always active)
  - 32 query heads / 8 KV heads (GQA), head_dim = 2048/32 = 64
  - vocab_size=128_000, max_position_embeddings=4096
  - Activation: SiLU, Norm: RMSNorm, dtype: bfloat16
  - trust_remote_code=True required (custom param2moe modeling code)
  - Thinking model: generates <think>...</think> before final answer
  - skip_special_tokens=False required when decoding (thinking tags are special tokens)
  - parsers.py ships inside the model repo (downloaded with snapshot_download)

TurboQuant on Param2:
  - 8 KV heads × 21 layers × head_dim=64 → very small KV footprint already
  - TurboQuant still helps: at batch_size>1 or long-context the savings compound
  - Protected layers (first+last 2) are especially important for MoE because
    the first layer's dense pre-MoE transform and last layer's lm_head projection
    are quality-critical
  - residual_window=128 is safe; Param2 max_ctx=4096 so no overflow risk

VRAM estimate (bf16, batch=1):
  Model weights:  ~34 GB  (17B × 2 bytes)
  KV cache bf16:  21 layers × 8 heads × 4096 × 64 × 2 × 2B ≈ 180 MB
  KV cache TQ:    ~180 MB / 70× ≈ 2.6 MB  (same ratio as Falcon)

Multi-GPU:
  34 GB doesn't fit on a single 24 GB GPU → device_map="auto" required
  Two 24 GB GPUs (e.g. 2× RTX 3090) are the minimum
  One 40 GB A100 fits with ~6 GB headroom
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------
# Param2-Thinking generates:
#   <think>...reasoning...</think>
#   <tool_call>...JSON...</tool_call>   (optional)
#   final answer text
#
# The model card ships a parsers.py inside the HF repo; we replicate its
# logic here so users don't need a separate download step.

_THINK_RE     = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@dataclass
class Param2Output:
    """Structured output from Param2-Thinking."""
    reasoning:    str         # content inside <think>...</think>
    tool_calls:   List[str]   # raw JSON strings from <tool_call> blocks
    final_answer: str         # text after all special blocks
    raw:          str         # full unmodified model output


def parse_model_output(text: str) -> Param2Output:
    """
    Parse raw Param2-Thinking output into structured components.

    Replicates the logic of the official parsers.py shipped in the model repo.
    Always call tokenizer.decode(..., skip_special_tokens=False) before passing
    text here — the thinking tags are special tokens and must be preserved.

    Args:
        text: Raw decoded model output (skip_special_tokens=False).

    Returns:
        Param2Output with reasoning, tool_calls, and final_answer fields.
    """
    # Extract reasoning block
    think_match = _THINK_RE.search(text)
    reasoning   = think_match.group(1).strip() if think_match else ""

    # Extract tool call blocks
    tool_calls = [m.group(1).strip() for m in _TOOL_CALL_RE.finditer(text)]

    # Final answer = everything after last closing tag
    remainder = text
    for pattern in (_THINK_RE, _TOOL_CALL_RE):
        remainder = pattern.sub("", remainder)
    final_answer = remainder.strip()

    return Param2Output(
        reasoning=reasoning,
        tool_calls=tool_calls,
        final_answer=final_answer,
        raw=text,
    )


def parse_dict(text: str) -> Dict[str, object]:
    """Dict-style output compatible with the official parsers.parse_model_output."""
    out = parse_model_output(text)
    return {
        "reasoning":    out.reasoning,
        "tool_calls":   out.tool_calls,
        "final_answer": out.final_answer,
    }


# ---------------------------------------------------------------------------
# Param2 config helpers
# ---------------------------------------------------------------------------

# Known architecture constants — used when the HF config isn't yet loaded
# (e.g. to pre-size the KV cache before model.load())
PARAM2_ARCH = dict(
    num_hidden_layers         = 21,
    hidden_size               = 2048,
    num_attention_heads       = 32,
    num_key_value_heads       = 8,
    head_dim                  = 64,       # hidden_size / num_attention_heads
    max_position_embeddings   = 4096,
    vocab_size                = 128_000,
    num_local_experts         = 64,
    num_shared_experts        = 2,
    num_experts_per_tok       = 6,
    intermediate_size         = 7168,
)

# Recommended TurboQuant settings for Param2
# Lower residual_window than Falcon because MoE layers are more sensitive
# to quantization error (the router uses the hidden state directly).
PARAM2_TURBOQUANT_DEFAULTS = dict(
    key_bits          = 4,
    value_bits        = 2,
    residual_window   = 128,
    protected_layers  = 2,   # first + last 2 layers at higher precision
)

# Recommended generation params from the model card
PARAM2_GENERATION_DEFAULTS = dict(
    temperature       = 0.7,
    top_k             = 50,
    top_p             = 0.9,
    do_sample         = True,
    max_new_tokens    = 512,
)


def build_param2_config(
    model_name: str = "bharatgenai/Param2-17B-A2.4B-Thinking",
    kv_cache:   str = "turboquant",
    device:     str = "auto",
    **generation_overrides,
) -> "bhaskera.config.Config":  # type: ignore[name-defined]
    """
    Build a Bhaskera Config pre-tuned for Param2-17B-A2.4B-Thinking.

    Args:
        model_name:  HF model id (default: official Param2 Thinking checkpoint).
        kv_cache:    "turboquant" | "static" | "none".
        device:      "auto" | "cuda" | "cpu".
        **generation_overrides: Override any PARAM2_GENERATION_DEFAULTS key.

    Returns:
        A fully populated Config ready to pass to InferenceEngine.

    Example::

        from bhaskera.inference.param2 import build_param2_config
        from bhaskera.inference import InferenceEngine

        cfg    = build_param2_config(kv_cache="turboquant")
        engine = InferenceEngine(cfg)
        engine.load()
        outputs = engine.generate_param2(["Explain attention mechanisms."])
        print(outputs[0].final_answer)
    """
    from bhaskera.config import (
        Config, ModelConfig, InferenceConfig,
        TurboQuantConfig, SpeculativeConfig,
    )

    gen = {**PARAM2_GENERATION_DEFAULTS, **generation_overrides}

    tq = TurboQuantConfig(**PARAM2_TURBOQUANT_DEFAULTS)

    return Config(
        model=ModelConfig(
            name=model_name,
            dtype="bfloat16",
            attn_impl=None,
            trust_remote_code=True,   # REQUIRED for param2moe custom code
        ),
        inference=InferenceConfig(
            max_new_tokens = gen["max_new_tokens"],
            temperature    = gen["temperature"],
            top_p          = gen["top_p"],
            top_k          = gen["top_k"],
            do_sample      = gen["do_sample"],
            batch_size     = 1,
            kv_cache       = kv_cache,
            device         = device,
            torch_compile  = False,   # custom forward — compile risky
            turboquant     = tq,
            speculative    = SpeculativeConfig(enabled=False),
        ),
    )


# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------

def apply_param2_chat_template(
    tokenizer,
    messages: List[Dict[str, str]],
    system_prompt: str = "You are a helpful assistant.",
) -> "torch.Tensor":  # type: ignore[name-defined]
    """
    Apply Param2's chat template to a list of messages.

    Args:
        tokenizer: Loaded Param2 tokenizer.
        messages:  List of {"role": "user"|"assistant", "content": "..."}.
        system_prompt: System message injected at position 0.

    Returns:
        input_ids tensor ready for model.generate().

    Notes:
        - Param2 uses apply_chat_template (standard HF interface).
        - skip_special_tokens=False is REQUIRED on the output side.
        - The tokenizer must be loaded with trust_remote_code=False
          (the tokenizer is standard; only the model needs trust_remote_code).
    """
    import torch

    conversation = [{"role": "system", "content": system_prompt}] + messages

    input_ids = tokenizer.apply_chat_template(
        conversation=conversation,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    return input_ids


# ---------------------------------------------------------------------------
# Thinking-aware stream decoder
# ---------------------------------------------------------------------------

class Param2StreamDecoder:
    """
    Incremental decoder that separates thinking tokens from answer tokens
    in a streaming generation loop.

    Usage::

        decoder = Param2StreamDecoder(tokenizer)
        for token_id in stream:
            event = decoder.step(token_id)
            if event.type == "thinking":
                print(".", end="", flush=True)   # progress dot
            elif event.type == "answer":
                print(event.text, end="", flush=True)
    """

    @dataclass
    class Event:
        type: str    # "thinking" | "answer" | "tool_call" | "eos"
        text: str

    _THINK_OPEN_STR  = "<think>"
    _THINK_CLOSE_STR = "</think>"
    _TOOL_OPEN_STR   = "<tool_call>"
    _TOOL_CLOSE_STR  = "</tool_call>"

    def __init__(self, tokenizer):
        self.tokenizer    = tokenizer
        self._buffer      = []
        self._in_thinking = False
        self._in_tool     = False
        self._full_ids    = []

    def step(self, token_id: int) -> "Param2StreamDecoder.Event":
        self._full_ids.append(token_id)
        token_str = self.tokenizer.decode([token_id], skip_special_tokens=False)

        # EOS
        if token_id == self.tokenizer.eos_token_id:
            return self.Event(type="eos", text="")

        # State machine: detect think/tool open/close tags
        self._buffer.append(token_str)
        joined = "".join(self._buffer)

        if not self._in_thinking and not self._in_tool:
            if self._THINK_OPEN_STR in joined:
                self._in_thinking = True
                self._buffer = []
                return self.Event(type="thinking", text="")
            if self._TOOL_OPEN_STR in joined:
                self._in_tool = True
                self._buffer = []
                return self.Event(type="tool_call", text="")
            # Regular answer token
            text = joined
            self._buffer = []
            return self.Event(type="answer", text=text)

        elif self._in_thinking:
            if self._THINK_CLOSE_STR in joined:
                self._in_thinking = False
                self._buffer = []
            return self.Event(type="thinking", text=token_str)

        else:  # in_tool
            if self._TOOL_CLOSE_STR in joined:
                self._in_tool = False
                self._buffer = []
            return self.Event(type="tool_call", text=token_str)

    def get_full_text(self) -> str:
        return self.tokenizer.decode(self._full_ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------
# Public re-exports for convenience
# ---------------------------------------------------------------------------
__all__ = [
    "Param2Output",
    "parse_model_output",
    "parse_dict",
    "build_param2_config",
    "apply_param2_chat_template",
    "Param2StreamDecoder",
    "PARAM2_ARCH",
    "PARAM2_TURBOQUANT_DEFAULTS",
    "PARAM2_GENERATION_DEFAULTS",
]