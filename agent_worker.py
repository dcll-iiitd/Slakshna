"""Persistent agent worker (A1).

Unlike ml_engine.py, which runs once per epoch and prints one JSON line, this
process stays alive and serves requests from the Rust node over local RPC:
one JSON object per line on stdin, one JSON reply per line on stdout.

    request : {"id": <int>, "method": <str>, "params": {...}}
    reply   : {"id": <int>, "ok": true,  "result": {...}}
              {"id": <int>, "ok": false, "error": "..."}

Diagnostics go to stderr; stdout carries only protocol replies.

Methods
    ping      -> {"pid": ..., "uptime_s": ...}
    chat      -> forwards an OpenAI-compatible chat completion to the federation
                 serving endpoint (SLAKSHNA_SERVE_URL; model from SLAKSHNA_SERVE_MODEL or
                 the endpoint's /v1/models) and returns the text plus
                 the token usage the server reported, so the node can charge
                 the agent's budget.
    shutdown  -> replies, then exits
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

START = time.time()
MAX_REPLY_BYTES = 8 * 1024 * 1024


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def do_ping(_params):
    return {"pid": os.getpid(), "uptime_s": time.time() - START}


_MODEL_CACHE = {}


def served_model(url, timeout=10):
    """The model to ask for: SLAKSHNA_SERVE_MODEL, else the first model the endpoint lists (cached).
    A real OpenAI-compatible server (vLLM) rejects an unknown model name with 404."""
    explicit = os.environ.get("SLAKSHNA_SERVE_MODEL")
    if explicit:
        return explicit
    if url not in _MODEL_CACHE:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=timeout) as resp:
                data = json.loads(resp.read(1 << 20))
            _MODEL_CACHE[url] = data["data"][0]["id"]
        except Exception:
            return "default"  # servers that ignore the field (and test doubles) still work
    return _MODEL_CACHE[url]


def do_chat(params):
    url = params.get("url") or os.environ.get("SLAKSHNA_SERVE_URL")
    if not url:
        raise ValueError("no serving endpoint: set SLAKSHNA_SERVE_URL or pass 'url'")
    body = {
        "model": params.get("model") or served_model(url),
        "messages": params["messages"],
        "max_tokens": int(params.get("max_tokens", 256)),
        "temperature": float(params.get("temperature", 0.0)),
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(params.get("timeout_s", 60))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(MAX_REPLY_BYTES + 1)
    except urllib.error.HTTPError as e:
        try:
            detail = e.read(400).decode("utf-8", "replace").replace("\n", " ")
        except Exception:
            detail = ""
        raise RuntimeError(f"serving endpoint returned HTTP {e.code}: {detail}" if detail else f"serving endpoint returned HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"serving endpoint unreachable: {e.reason}") from e
    if len(raw) > MAX_REPLY_BYTES:
        raise RuntimeError("serving reply too large")
    data = json.loads(raw)
    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    total = usage.get("total_tokens")
    if total is None:
        total = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    return {"text": choice, "total_tokens": int(total)}


METHODS = {"ping": do_ping, "chat": do_chat}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            rid = req["id"]
        except Exception as e:  # malformed request: nothing to correlate a reply with
            log(f"bad request: {e}")
            continue
        method = req.get("method")
        if method == "shutdown":
            print(json.dumps({"id": rid, "ok": True, "result": {}}), flush=True)
            return
        try:
            fn = METHODS.get(method)
            if fn is None:
                raise ValueError(f"unknown method '{method}'")
            reply = {"id": rid, "ok": True, "result": fn(req.get("params") or {})}
        except Exception as e:
            reply = {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
