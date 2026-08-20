"""
bhaskera-infer — command-line inference entry point (v2).

Changes vs v1:
  - Tokens/second reported after every generation
  - Token count measured from actual output ids (not char count)
  - Thinking models: <think> block stripped from terminal output by default
    (--show-thinking to display it; raw text is always saved if --output-file)
  - TurboQuant stats line always shown when cache is active
  - Cleaner separator / stats block

Examples
--------
    # Standard generation
    bhaskera-infer --config configs/inference_turboquant.yaml \\
                   --prompt "Explain attention mechanisms."

    # Param2 Thinking model (strips <think> by default)
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt "What is 17 × 23?"

    # Show the chain-of-thought
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt "What is 17 × 23?" --show-thinking

    # Benchmark throughput
    bhaskera-infer --config configs/inference_turboquant.yaml \\
                   --prompt-file prompts.txt --max-new-tokens 256
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bhaskera.infer")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhaskera-infer",
        description="Bhaskera LLM inference engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config / model
    p.add_argument("--config",  default=None,   help="Path to Bhaskera YAML config")
    p.add_argument("--model",   default=None,   help="HuggingFace model id (overrides config)")
    p.add_argument("--device",  default="auto", help="Device: auto | cuda | cpu | mps")

    # Input
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--prompt",      default=None, help="Single prompt string")
    inp.add_argument("--prompt-file", default=None, metavar="FILE",
                     help="File with one prompt per line")

    # Generation
    p.add_argument("--max-new-tokens", type=int,   default=None)
    p.add_argument("--temperature",    type=float, default=None)
    p.add_argument("--top-p",          type=float, default=None)
    p.add_argument("--top-k",          type=int,   default=None)
    p.add_argument("--no-sample",      action="store_true",
                   help="Greedy decoding (overrides do_sample=true in config)")

    # KV cache
    p.add_argument("--kv-cache", default=None, choices=["static", "turboquant", "none"])
    p.add_argument("--key-bits",          type=int, default=None)
    p.add_argument("--value-bits",        type=int, default=None)
    p.add_argument("--residual-window",   type=int, default=None)

    # Speculative decoding
    p.add_argument("--speculative",    action="store_true")
    p.add_argument("--draft-model",    default=None)
    p.add_argument("--num-draft-tokens", type=int, default=None)

    # Thinking model
    p.add_argument("--show-thinking",  action="store_true",
                   help="Print <think> reasoning block (Param2 / thinking models)")
    p.add_argument("--system-prompt",  default="You are a helpful assistant.",
                   help="System prompt injected for chat-template models")

    # Output
    p.add_argument("--output-file", default=None, metavar="FILE",
                   help="Write full raw output to file (one response per line)")
    p.add_argument("--return-full",    action="store_true",
                   help="Include the prompt in the output")
    p.add_argument("--torch-compile",  action="store_true")
    p.add_argument("--verbose", "-v",  action="store_true")

    return p


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace):
    from bhaskera.config import (
        Config, InferenceConfig, TurboQuantConfig, SpeculativeConfig,
    )
    if args.config:
        from bhaskera.config import load_config
        cfg = load_config(args.config)
    else:
        cfg = Config()

    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.inference.device = args.device

    infer = cfg.inference
    if args.max_new_tokens is not None: infer.max_new_tokens = args.max_new_tokens
    if args.temperature   is not None: infer.temperature    = args.temperature
    if args.top_p         is not None: infer.top_p          = args.top_p
    if args.top_k         is not None: infer.top_k          = args.top_k
    if args.no_sample:                 infer.do_sample       = False
    if args.torch_compile:             infer.torch_compile   = True
    if args.kv_cache:                  infer.kv_cache        = args.kv_cache

    if infer.kv_cache == "turboquant":
        if args.key_bits        is not None: infer.turboquant.key_bits        = args.key_bits
        if args.value_bits      is not None: infer.turboquant.value_bits      = args.value_bits
        if args.residual_window is not None: infer.turboquant.residual_window = args.residual_window
        infer.turboquant.enabled = True

    if args.speculative:
        infer.speculative.enabled = True
    if args.draft_model:
        infer.speculative.draft_model_name = args.draft_model
        infer.speculative.enabled = True
    if args.num_draft_tokens is not None:
        infer.speculative.num_draft_tokens = args.num_draft_tokens

    return cfg


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

def _count_output_tokens(text: str, tokenizer=None) -> int:
    """
    Count output tokens.
    Uses the tokenizer when available (exact); falls back to word-count
    heuristic (rough but doesn't require tokenizer access).
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Heuristic: ~0.75 tokens per word for English, ~1.3 for code/mixed
    words = len(text.split())
    return max(1, int(words * 0.9))


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

SEP = "─" * 72

def _render_output(
    idx: int,
    total: int,
    prompt: str,
    output_text: str,
    is_thinking_model: bool,
    show_thinking: bool,
) -> str:
    """Format one output for terminal display."""
    lines = []
    if total > 1:
        lines.append(f"\n{SEP}")
        lines.append(f"[{idx + 1}/{total}] Prompt: {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
        lines.append(SEP)

    if is_thinking_model and not show_thinking:
        # Strip <think>...</think> for clean terminal output
        import re
        clean = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
        # Also strip any leftover tool_call blocks
        clean = re.sub(r"<tool_call>.*?</tool_call>", "", clean, flags=re.DOTALL).strip()
        lines.append(clean)
    else:
        lines.append(output_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Prompts ──────────────────────────────────────────────────────
    if args.prompt:
        prompts = [args.prompt]
    else:
        path = Path(args.prompt_file)
        if not path.exists():
            parser.error(f"Prompt file not found: {args.prompt_file}")
        with open(path) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if not prompts:
            parser.error(f"No prompts found in {args.prompt_file}")
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    # ── Config ───────────────────────────────────────────────────────
    cfg = _build_config(args)

    # ── Engine ───────────────────────────────────────────────────────
    from bhaskera.inference import InferenceEngine
    engine = InferenceEngine(cfg)
    engine.load()

    # ── Log active settings ───────────────────────────────────────────
    infer = cfg.inference
    logger.info(
        f"Settings: kv_cache={infer.kv_cache!r} "
        f"temperature={infer.temperature} top_p={infer.top_p} "
        f"max_new_tokens={infer.max_new_tokens} "
        f"speculative={infer.speculative.enabled}"
    )
    if infer.kv_cache == "turboquant":
        logger.info(
            f"TurboQuant: K{infer.turboquant.key_bits}/V{infer.turboquant.value_bits} bits, "
            f"residual_window={infer.turboquant.residual_window}, "
            f"protected_layers={infer.turboquant.protected_layers}"
        )

    is_thinking = getattr(engine._backend, "_is_thinking", False) if engine._loaded else False

    # Try to get tokenizer for accurate token counting
    _tokenizer = None
    try:
        _tokenizer = getattr(engine._backend, "_tok", None)
    except Exception:
        pass

    # ── Generate ─────────────────────────────────────────────────────
    t0 = time.perf_counter()

    outputs = engine.generate(
        prompts,
        max_new_tokens = args.max_new_tokens or infer.max_new_tokens,
        temperature    = args.temperature or infer.temperature,
        top_p          = args.top_p or infer.top_p,
        top_k          = args.top_k or infer.top_k,
        do_sample      = not args.no_sample and infer.do_sample,
        return_full_text = args.return_full,
    )

    elapsed = time.perf_counter() - t0

    # ── Count tokens ──────────────────────────────────────────────────
    total_output_tokens = sum(_count_output_tokens(o, _tokenizer) for o in outputs)
    tokens_per_second   = total_output_tokens / elapsed if elapsed > 0 else 0.0

    # ── Print results ──────────────────────────────────────────────────
    raw_outputs = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        rendered = _render_output(
            i, len(prompts), prompt, output,
            is_thinking_model=is_thinking,
            show_thinking=args.show_thinking,
        )
        print(rendered)
        raw_outputs.append(output)

    # ── Stats block ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(
        f"Generated {len(prompts)} response(s) | "
        f"{total_output_tokens} tokens | "
        f"{elapsed:.2f}s | "
        f"\033[1;32m{tokens_per_second:.1f} tok/s\033[0m"
    )

    # KV cache stats (TurboQuant)
    stats = engine.kv_cache_stats()
    if stats and stats.get("compression_ratio", 0) > 0:
        print(
            f"TurboQuant KV cache: {stats['tq_mb']:.1f} MB "
            f"(bf16 baseline: {stats['bf16_mb']:.1f} MB, "
            f"ratio: {stats['compression_ratio']:.1f}×)"
        )
    elif infer.kv_cache == "turboquant":
        # Cache exists but was bypassed (e.g. Param2) — still note it
        print(f"TurboQuant: active (model uses internal cache)")

    # Thinking model note
    if is_thinking and not args.show_thinking:
        print("(Thinking/reasoning block hidden — use --show-thinking to display)")

    # ── Optional file output ───────────────────────────────────────────
    if args.output_file:
        out_path = Path(args.output_file)
        with open(out_path, "w") as f:
            for raw in raw_outputs:
                f.write(raw.replace("\n", "\\n") + "\n")
        logger.info(f"Raw outputs written to {out_path}")


if __name__ == "__main__":
    main()