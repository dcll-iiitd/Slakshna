"""
Local inference sidecar for federated inference (step 6).

Spawned by the Rust node (``[inference] spawn_sidecar = true``) under the
Bhaskera *serve* environment. It loads the base model once through the
``Bhaskera-serve`` submodule's ``HFEngineWrapper`` and hot-swaps PEFT LoRA
adapters per request, so the full forward pass runs on this machine.

Protocol (loopback HTTP, JSON):
    GET  /health
    POST /generate {prompt, max_new_tokens, temperature, adapter_id?, adapter_dir?}
         -> {text, adapter_id}
    POST /decide {state, question, options[], adapters: [{id, dir, weight}], combine?,
                  permutations?, calibration_temperature?}
         -> typed System One decision: probabilities over `options`, confidence, per-adapter
            distributions (see federated_inference/system_one.py). One forward pass per adapter, no generation.

The Rust node has already selected, downloaded and SHA-256-verified the adapter
directory before calling /generate; this process never touches the network.

Environment:
    SLAKSHNA_INFER_PORT        port to listen on (default 8765)
    SLAKSHNA_INFER_BASE_MODEL  HF model name or local path
    SLAKSHNA_SERVE_PATH        path to the Bhaskera-serve checkout
    SLAKSHNA_INFER_MAX_ADAPTERS  adapters kept loaded at once (default 4)
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

    def decide(self, state: str, question: str, options: list[str], members: list[dict],
               combine: str, permutations: int, calibration_temperature: float) -> dict:
        """System One decision: each adapter scores the closed option set in one pass."""
        import torch
        from federated_inference import system_one as so

        if len(members) > self._max_adapters:
            raise ValueError(f"{len(members)} adapters exceed SLAKSHNA_INFER_MAX_ADAPTERS={self._max_adapters}")
        tok = self._engine.get_tokenizer()
        device = self._engine._device

        with self._lock:
            for m in members:
                self._attach(m["id"], m["dir"])
            model = self._peft if self._peft is not None else self._engine._model
            max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)

            stats = {"forward_passes": 0, "model_s": 0.0}

            def logits_fn_for(name):
                def fn(prompt: str):
                    t0 = time.perf_counter()
                    try:
                        return _fn(prompt)
                    finally:
                        stats["forward_passes"] += 1
                        stats["model_s"] += time.perf_counter() - t0

                def _fn(prompt: str):
                    ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
                    if max_pos and ids.shape[1] > max_pos:
                        raise ValueError(f"prompt is {ids.shape[1]} tokens, model limit is {max_pos}")
                    with torch.no_grad():
                        if name is not None:
                            self._peft.set_adapter(name)
                            out = model(input_ids=ids)
                        elif self._peft is not None:
                            with self._peft.disable_adapter():
                                out = model(input_ids=ids)
                        else:
                            out = model(input_ids=ids)
                    return out.logits[0, -1].float()
                return fn

            if members:
                mem = [so.Member(m["id"], logits_fn_for(m["id"]), float(m["weight"])) for m in members]
            else:
                mem = [so.Member("base", logits_fn_for(None), 1.0)]
            decision = so.decide(
                mem, tok, state, question, options, rule=combine,
                permutations=permutations, calibration_temperature=calibration_temperature,
            )
        out = decision.to_dict()
        out["forward_passes"] = stats["forward_passes"]
        out["model_ms"] = stats["model_s"] * 1000.0
        return out

    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 adapter_id: str | None, adapter_dir: str | None) -> str:
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
            if self.path not in ("/generate", "/decide"):
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    return self._send(400, {"error": "bad body size"})
                req = json.loads(self.rfile.read(length))
                if self.path == "/decide":
                    members = req.get("adapters") or []
                    if not isinstance(members, list) or len(members) > 16:
                        return self._send(400, {"error": "adapters must be a list of at most 16"})
                    for m in members:
                        d = Path(str(m["dir"]))
                        if not (d / "adapter_model.safetensors").is_file() or not (d / "adapter_config.json").is_file():
                            return self._send(400, {"error": "adapter dir is not a PEFT adapter directory"})
                    combine_rule = str(req.get("combine", "prob_mean"))
                    if combine_rule not in ("prob_mean", "logit_mean"):
                        return self._send(400, {"error": "unknown combine rule"})
                    options = req["options"]
                    if not isinstance(options, list) or not all(isinstance(o, str) for o in options):
                        return self._send(400, {"error": "options must be a list of strings"})
                    out = model.decide(
                        str(req.get("state", "")), str(req["question"]), options, members,
                        combine_rule,
                        max(0, min(int(req.get("permutations", 0)), 8)),  # 0 = auto
                        float(req.get("calibration_temperature", 1.0)),
                    )
                    return self._send(200, out)
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
                )
                self._send(200, {"text": text, "adapter_id": adapter_id})
            except ValueError as e:
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
