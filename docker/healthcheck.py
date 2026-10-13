#!/usr/bin/env python
"""Docker HEALTHCHECK: GET /status on the node's API port (no curl needed)."""
import os
import sys
import tomllib
import urllib.request

try:
    with open(os.environ.get("SLAKSHNA_CONFIG", "/app/config.toml"), "rb") as f:
        net = tomllib.load(f).get("network", {})
    host = net.get("host", "127.0.0.1")
    if host in ("0.0.0.0", ""):
        host = "127.0.0.1"
    url = f"http://{host}:{net['api_port']}/status"
    with urllib.request.urlopen(url, timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception as e:  # noqa: BLE001
    print(f"unhealthy: {e}", file=sys.stderr)
    sys.exit(1)
