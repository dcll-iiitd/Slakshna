#!/usr/bin/env bash
# Container entrypoint: preflight checks, then replace this shell with the node.
set -euo pipefail

CONFIG="${SLAKSHNA_CONFIG:-/app/config.toml}"

if [ "${SLAKSHNA_SKIP_PREFLIGHT:-0}" != "1" ]; then
    python /usr/local/bin/preflight.py "$CONFIG"
fi

# exec: the node becomes PID 1's child (under tini when compose sets init: true),
# so SIGTERM from `docker stop` reaches it directly.
exec iiitd --config "$CONFIG" "$@"
