"""Write the configs for N RAG nodes on one machine that all know each other.

    python3 scripts/rag_make_configs.py --nodes 5 --name book \
        --out m9-configs/book --data-root m9-data/book --embedding-model BAAI/bge-base-en-v1.5

Each node gets its own ports, data_dir and sidecar port; discovery is off and
`network.peers` lists every other node as <EndpointId>@127.0.0.1:<port>, so
every node asks all N-1 others. EndpointIds only
exist once a node has an identity, so the configs are written twice: first
without peers, then each node's identity is created with
`iiitd --print-endpoint-id`, then the peers are filled in. The identity lives
in data_dir, so re-running keeps the same ids.

Also writes nodes.json (index, EndpointId, API URL, adversary) for the
cluster script and the evaluation.
`--adversary` makes one node misbehave (experiments only, src/rag/adversary.rs).
Standard library only.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ADVERSARIES = ["none", "tamper_text", "inflate_score", "fake_centroid", "hide_cluster", "equivocate"]


def toml_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    return json.dumps(value)  # a TOML basic string has the same escapes as JSON


def render(sections):
    lines = ["# Written by scripts/rag_make_configs.py; edit that script, not this file.", ""]
    for name, values in sections.items():
        lines.append(f"[{name}]")
        lines.extend(f"{key} = {toml_value(value)}" for key, value in values.items())
        lines.append("")
    return "\n".join(lines)


def node_config(args, i, peers, allowed):
    return {
        "federation": {"id": args.federation_id or f"slakshna-m9-{args.name}", "name": f"SLAKSHNA M9 {args.name}"},
        "training": {"enabled": False},
        "node": {"id": f"m9-{args.name}-node{i}", "data_dir": os.path.join(args.data_root, f"node{i}")},
        "network": {
            "host": "127.0.0.1",  # the HTTP API stays on this machine
            "p2p_port": args.p2p_port + i,
            "api_port": args.api_port + i,
            "ws_port": args.ws_port + i,
            "peers": peers,
            "allowed_peers": allowed,
        },
        "rag": {
            "enabled": True,
            "sidecar_port": args.sidecar_port + i,
            "python": args.python,
            "embedding_model": args.embedding_model,
            "generator": "none",  # contexts only; answers come from scripts/m9/answer.py
            "top_k": args.k,
            "final_k": args.k,
            "max_clusters": args.max_clusters,
            "chunks_per_cluster": args.chunks_per_cluster,
            "select_timeout_ms": args.select_timeout_ms,
            "peer_timeout_ms": args.peer_timeout_ms,
            "max_message_bytes": args.max_message_bytes,
            "directory": True,
            "adversary": args.adversary if i == args.adversary_node else "none",
        },
        "discovery": {"mdns": False, "dht": False, "dns": False, "relay": False},
        "logging": {"level": "info"},
    }


def write_configs(args, peers_of, allowed):
    paths = []
    for i in range(args.nodes):
        path = os.path.join(args.out, f"node{i}.toml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(node_config(args, i, peers_of(i), allowed)))
        paths.append(path)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=5)
    p.add_argument("--name", required=True, help="run name, e.g. the shard scheme; used in the federation id")
    p.add_argument("--out", required=True, help="folder for node<i>.toml and nodes.json")
    p.add_argument("--data-root", required=True, help="node i keeps its identity and index in <data-root>/node<i>")
    p.add_argument("--federation-id", help="default: slakshna-m9-<name>")
    p.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("--python", default=".venv-rag/bin/python")
    p.add_argument("--binary", default="target/release/iiitd")
    p.add_argument("--p2p-port", type=int, default=9100, help="node i uses this + i (UDP)")
    p.add_argument("--api-port", type=int, default=18545)
    p.add_argument("--ws-port", type=int, default=18645)
    p.add_argument("--sidecar-port", type=int, default=18600)
    p.add_argument("--k", type=int, default=8, help="top_k and final_k")
    p.add_argument("--max-clusters", type=int, default=4)
    p.add_argument("--chunks-per-cluster", type=int, default=8)
    p.add_argument("--select-timeout-ms", type=int, default=5000)
    p.add_argument("--peer-timeout-ms", type=int, default=15000)
    p.add_argument("--max-message-bytes", type=int, default=4 * 1024 * 1024)
    p.add_argument("--adversary", default="none", choices=ADVERSARIES)
    p.add_argument("--adversary-node", type=int, default=-1, help="index of the misbehaving node")
    args = p.parse_args()
    if args.adversary != "none" and not 0 <= args.adversary_node < args.nodes:
        p.error("--adversary needs --adversary-node in [0, nodes)")
    if args.adversary_node == 0:
        print("warning: node 0 is the one queries are sent to; an attacker there attacks only itself",
              file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    paths = write_configs(args, lambda i: [], [])
    ids = []
    for path in paths:
        result = subprocess.run([os.path.join(ROOT, args.binary) if not os.path.isabs(args.binary) else args.binary,
                                 "--config", path, "--print-endpoint-id"],
                                capture_output=True, text=True, cwd=ROOT)
        if result.returncode != 0:
            sys.exit(f"could not create the identity for {path}:\n{result.stderr}")
        ids.append(result.stdout.strip().splitlines()[-1])
    if len(set(ids)) != len(ids):
        sys.exit(f"two nodes got the same EndpointId ({ids}); do they share a data_dir?")

    write_configs(args, lambda i: [f"{ids[j]}@127.0.0.1:{args.p2p_port + j}" for j in range(args.nodes) if j != i], ids)
    nodes = [{
        "index": i, "config": paths[i], "endpoint_id": ids[i],
        "api": f"http://127.0.0.1:{args.api_port + i}",
        "data_dir": os.path.join(args.data_root, f"node{i}"),
        "adversary": args.adversary if i == args.adversary_node else "none",
    } for i in range(args.nodes)]
    with open(os.path.join(args.out, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "embedding_model": args.embedding_model, "nodes": nodes}, f, indent=2)
    for node in nodes:
        print(f"node{node['index']}: {node['endpoint_id']}  {node['api']}  adversary={node['adversary']}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
