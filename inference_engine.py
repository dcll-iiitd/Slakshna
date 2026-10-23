"""
Local inference sidecar for federated inference (step 6).

Spawned by the Rust node (``[inference] spawn_sidecar = true``) under the
Bhaskera *serve* environment. It loads the base model once through the
``Bhaskera-serve`` submodule's ``HFEngineWrapper`` and hot-swaps PEFT LoRA
adapters per request, so the full forward pass runs on this machine.

Protocol (loopback HTTP, JSON):
    GET  /health
    POST /generate {prompt, max_new_tokens, temperature, adapter_id?, adapter_dir?, chat?, system?}
         -> {text, adapter_id}
         chat (default true): wrap the prompt in the model's chat template, because the
         adapters were trained on chat-formatted data; a chat model fed a bare prompt tends
         to continue it ("What is LoRA?" -> " What is it used for?") instead of answering.

The Rust node has already selected, downloaded and SHA-256-verified the adapter
directory before calling /generate; this process never touches the network.

Environment:
    SLAKSHNA_INFER_PORT        port to listen on (default 8765)
    SLAKSHNA_INFER_BASE_MODEL  HF model name or local path
    SLAKSHNA_SERVE_PATH        path to the Bhaskera-serve checkout
    SLAKSHNA_INFER_MAX_ADAPTERS  adapters kept loaded at once (default 4)
    SLAKSHNA_INFER_MAX_NEW_TOKENS  hard cap per request (default 2048)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_NEW_TOKENS = int(os.environ.get("SLAKSHNA_INFER_MAX_NEW_TOKENS", "2048"))


def format_prompt(tok, prompt: str, chat: bool = True, system: str | None = None) -> str:
    """The text actually fed to the model. With a chat template (and chat=True) the
    prompt becomes a user turn followed by the assistant-turn opener."""
    if chat and getattr(tok, "chat_template", None):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def log(msg: str) -> None:
    print(f"[inference_engine] {msg}", file=sys.stderr, flush=True)


def _add_serve_to_path() -> None:
    serve_path = Path(os.environ.get("SLAKSHNA_SERVE_PATH", "Bhaskera-serve")).resolve() / "src"
    if not serve_path.is_dir():
        raise SystemExit(f"Bhaskera-serve not found at {serve_path}; run `git submodule update --init Bhaskera-serve`")
    sys.path.insert(0, str(serve_path))


class AdapterModel:
    """Base model plus a small LRU of attached PEFT adapters."""

    def __init__(self, base_model: str, max_adapters: int) -> None:
        _add_serve_to_path()
        from bhaskera.config import load_config
        from bhaskera.serve.engine import GenerationParams, HFEngineWrapper

        self._GenerationParams = GenerationParams
        # Minimal serve config: plain base model, no training-time LoRA.
        cfg_yaml = (
            f"model:\n  name: {json.dumps(base_model)}\n  dtype: bfloat16\n"
            "serve:\n  backend: hf\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(cfg_yaml)
            cfg_path = f.name
        try:
            self._engine = HFEngineWrapper(load_config(cfg_path))
        finally:
            os.unlink(cfg_path)

        self._lock = threading.Lock()  # model.generate() is not thread-safe
        self._peft = None  # PeftModel once the first adapter is attached
        self._loaded: "OrderedDict[str, str]" = OrderedDict()  # adapter_id -> dir
        self._max_adapters = max_adapters

    def _attach(self, adapter_id: str, adapter_dir: str) -> None:
        from peft import PeftModel

        if adapter_id in self._loaded:
            self._loaded.move_to_end(adapter_id)
            return
        if self._peft is None:
            self._peft = PeftModel.from_pretrained(
                self._engine._model, adapter_dir, adapter_name=adapter_id, is_trainable=False
            )
            self._peft.eval()
            self._engine._model = self._peft
        else:
            self._peft.load_adapter(adapter_dir, adapter_name=adapter_id, is_trainable=False)
        self._loaded[adapter_id] = adapter_dir
        log(f"attached adapter {adapter_id[:12]} ({len(self._loaded)} loaded)")

        while len(self._loaded) > self._max_adapters:
            victim, _ = next(iter(self._loaded.items()))
            if victim == adapter_id:
                break
            self._peft.delete_adapter(victim)
            del self._loaded[victim]
            log(f"evicted adapter {victim[:12]}")

    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 adapter_id: str | None, adapter_dir: str | None,
                 chat: bool = True, system: str | None = None, min_new_tokens: int = 0):
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS))
        tok = self._engine.get_tokenizer()
        prompt = format_prompt(tok, prompt, chat, system)
        limit = getattr(getattr(self._engine._model, "config", None), "max_position_embeddings", None)
        if limit:
            n_prompt = len(tok(prompt)["input_ids"])
            if n_prompt + max_new_tokens > limit:
                raise ValueError(f"prompt ({n_prompt} tokens) plus max_new_tokens ({max_new_tokens}) exceeds the model limit of {limit}")
        params = self._GenerationParams(max_new_tokens=max_new_tokens, temperature=temperature)

        min_new_tokens = max(0, min(int(min_new_tokens or 0), max_new_tokens))

        async def fixed_length_stream():
            # Same streaming as the engine, plus min_new_tokens so benchmarks can force an exact
            # output length (TPOT and throughput need long, equal-length generations).
            import torch
            from transformers import TextIteratorStreamer

            mtok = self._engine._tokenizer
            inputs = mtok(prompt, return_tensors="pt").to(self._engine._device)
            streamer = TextIteratorStreamer(mtok, skip_prompt=True, skip_special_tokens=True, timeout=None)
            kw = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                      do_sample=temperature > 0.0, temperature=max(temperature, 1e-6),
                      pad_token_id=mtok.pad_token_id, eos_token_id=mtok.eos_token_id)
            threading.Thread(target=lambda: torch.inference_mode()(self._engine._model.generate)(**kw), daemon=True).start()
            done = object()

            def nxt():
                try:
                    return next(streamer)
                except StopIteration:
                    return done

            while True:
                chunk = await asyncio.to_thread(nxt)
                if chunk is done:
                    break
                if chunk:
                    yield chunk

        async def timed():
            # Streams the answer so time-to-first-token and time-per-output-token can be measured.
            t0 = time.perf_counter()
            first = None
            parts: list[str] = []
            stream = fixed_length_stream() if min_new_tokens > 0 else self._engine.generate(prompt, params)
            async for chunk in stream:
                if chunk and first is None:
                    first = time.perf_counter()
                parts.append(chunk)
            t1 = time.perf_counter()
            text = "".join(parts)
            n = len(tok(text, add_special_tokens=False)["input_ids"]) if text else 0
            first = first if first is not None else t1
            return text, {
                "ttft_ms": (first - t0) * 1000.0,
                "tpot_ms": ((t1 - first) * 1000.0 / (n - 1)) if n > 1 else 0.0,
                "gen_ms": (t1 - t0) * 1000.0,
                "n_tokens": n,
                "n_prompt_tokens": len(tok(prompt)["input_ids"]),
            }

        with self._lock:
            attach_ms = 0.0
            if adapter_id and adapter_dir:
                t = time.perf_counter()
                self._attach(adapter_id, adapter_dir)
                self._peft.set_adapter(adapter_id)
                attach_ms = (time.perf_counter() - t) * 1000.0
                text, timing = asyncio.run(timed())
            elif self._peft is not None:
                with self._peft.disable_adapter():
                    text, timing = asyncio.run(timed())
            else:
                text, timing = asyncio.run(timed())
        timing["attach_ms"] = attach_ms
        return text, timing


def make_handler(model: AdapterModel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep stdout clean; stderr only
            log(fmt % args)

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path in ("/stats", "/stats/reset"):
                # GPU memory of this process (for the memory benchmark); /stats/reset also resets the peak.
                import torch
                st = {"allocated_mb": torch.cuda.memory_allocated() / 2**20,
                      "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                      "reserved_mb": torch.cuda.memory_reserved() / 2**20,
                      "adapters_loaded": len(model._loaded)}
                if self.path == "/stats/reset":
                    torch.cuda.reset_peak_memory_stats()
                self._send(200, st)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    return self._send(400, {"error": "bad body size"})
                req = json.loads(self.rfile.read(length))
                adapter_id = req.get("adapter_id")
                adapter_dir = req.get("adapter_dir")
                if adapter_dir:
                    d = Path(adapter_dir)
                    if not (d / "adapter_model.safetensors").is_file() or not (d / "adapter_config.json").is_file():
                        return self._send(400, {"error": "adapter_dir is not a PEFT adapter directory"})
                text, timing = model.generate(
                    str(req["prompt"]),
                    int(req.get("max_new_tokens", 256)),
                    float(req.get("temperature", 0.7)),
                    adapter_id,
                    adapter_dir,
                    bool(req.get("chat", True)),
                    req.get("system"),
                    int(req.get("min_new_tokens") or 0),
                )
                self._send(200, {"text": text, "adapter_id": adapter_id, "timing": timing})
            except ValueError as e:  # bad temperature, prompt too long, ...
                self._send(400, {"error": str(e)})
            except KeyError as e:
                self._send(400, {"error": f"missing field {e}"})
            except Exception as e:  # surfaced to the Rust node, which degrades gracefully
                log(f"generate failed: {e!r}")
                self._send(500, {"error": str(e)})

    return Handler


def main() -> None:
    base_model = os.environ.get("SLAKSHNA_INFER_BASE_MODEL", "")
    if not base_model:
        raise SystemExit("SLAKSHNA_INFER_BASE_MODEL is required")
    port = int(os.environ.get("SLAKSHNA_INFER_PORT", "8765"))
    max_adapters = int(os.environ.get("SLAKSHNA_INFER_MAX_ADAPTERS", "4"))

    log(f"loading base model {base_model} ...")
    model = AdapterModel(base_model, max_adapters)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(model))
    log(f"ready on 127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
