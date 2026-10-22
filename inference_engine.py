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
                 chat: bool = True, system: str | None = None) -> str:
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
        with self._lock:
            if adapter_id and adapter_dir:
                self._attach(adapter_id, adapter_dir)
                self._peft.set_adapter(adapter_id)
                return asyncio.run(self._engine.generate_full(prompt, params))
            if self._peft is not None:
                with self._peft.disable_adapter():
                    return asyncio.run(self._engine.generate_full(prompt, params))
            return asyncio.run(self._engine.generate_full(prompt, params))


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
                text = model.generate(
                    str(req["prompt"]),
                    int(req.get("max_new_tokens", 256)),
                    float(req.get("temperature", 0.7)),
                    adapter_id,
                    adapter_dir,
                    bool(req.get("chat", True)),
                    req.get("system"),
                )
                self._send(200, {"text": text, "adapter_id": adapter_id})
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
