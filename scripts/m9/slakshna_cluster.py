"""Start, wait for, and stop the Slakshna nodes described by a nodes.json.

    python3 scripts/m9/slakshna_cluster.py up   --configs m9-configs/book --threads 16
    python3 scripts/m9/slakshna_cluster.py wait --configs m9-configs/book
    python3 scripts/m9/slakshna_cluster.py down --configs m9-configs/book

`up` starts every node in the background (logs in <configs>/logs/), then
waits. A node is ready when its sidecar answers and, in its /rag/status,
every other node is a peer target whose signed root was fetched ("ok") and
whose centroid directory is complete. Run from the repository root.
Standard library only.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_nodes(configs):
    with open(os.path.join(configs, "nodes.json"), encoding="utf-8") as f:
        return json.load(f)["nodes"]


def pid_file(configs):
    return os.path.join(configs, "pids.json")


def status(api):
    with urllib.request.urlopen(f"{api}/rag/status", timeout=5) as response:
        return json.load(response)


def readiness(node, nodes):
    """(ready, short description) for one node."""
    try:
        s = status(node["api"])
    except Exception as e:  # not up yet
        return False, f"no API ({type(e).__name__})"
    others = {n["endpoint_id"] for n in nodes if n["index"] != node["index"]}
    targets = set(s.get("peer_targets", []))
    ok = {p["endpoint_id"] for p in s.get("registry", []) if p["status"] == "ok"}
    complete = {p["endpoint_id"] for p in s.get("directory", {}).get("peers", []) if p["complete"]}
    ready = others <= targets and others <= ok and others <= complete
    return ready, (f"targets {len(others & targets)}/{len(others)}, roots ok {len(others & ok)}, "
                   f"directories complete {len(others & complete)}")


def wait(configs, timeout, nodes=None):
    nodes = nodes or load_nodes(configs)
    deadline = time.time() + timeout
    while True:
        states = [readiness(node, nodes) for node in nodes]
        if all(ready for ready, _ in states):
            print(f"all {len(nodes)} nodes ready", file=sys.stderr)
            return True
        if time.time() > deadline:
            for node, (_, text) in zip(nodes, states):
                print(f"node{node['index']}: {text}", file=sys.stderr)
            print(f"not ready after {timeout} s; see {os.path.join(configs, 'logs')}", file=sys.stderr)
            return False
        print("  " + " | ".join(f"node{n['index']}: {t}" for n, (_, t) in zip(nodes, states)), file=sys.stderr)
        time.sleep(5)


def up(args):
    if os.path.exists(pid_file(args.configs)):
        sys.exit(f"{pid_file(args.configs)} exists: run 'down' first")
    nodes = load_nodes(args.configs)
    logs = os.path.join(args.configs, "logs")
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ)
    if args.threads:
        for name in ("SLAKSHNA_TORCH_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = str(args.threads)
    pids = {}
    for node in nodes:
        with open(os.path.join(logs, f"node{node['index']}.log"), "ab") as log:
            process = subprocess.Popen([args.binary, "--config", node["config"]], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pids[str(node["index"])] = process.pid
        print(f"node{node['index']}: pid {process.pid}, adversary={node['adversary']}", file=sys.stderr)
    with open(pid_file(args.configs), "w", encoding="utf-8") as f:
        json.dump(pids, f)
    if not wait(args.configs, args.timeout, nodes):
        sys.exit(1)


def down(args):
    try:
        with open(pid_file(args.configs), encoding="utf-8") as f:
            pids = json.load(f)
    except FileNotFoundError:
        print("no pids.json: nothing to stop", file=sys.stderr)
        return
    for pid in pids.values():
        try:
            os.killpg(pid, signal.SIGTERM)  # the node and its sidecar share a process group
        except ProcessLookupError:
            pass
    deadline = time.time() + 15
    while time.time() < deadline and any(alive(pid) for pid in pids.values()):
        time.sleep(0.5)
    for pid in pids.values():
        if alive(pid):
            os.killpg(pid, signal.SIGKILL)
    os.remove(pid_file(args.configs))
    print(f"stopped {len(pids)} nodes", file=sys.stderr)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A child that exited but wasn't reaped still has a pid; it isn't running.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["up", "wait", "down"])
    p.add_argument("--configs", required=True, help="folder written by scripts/rag_make_configs.py")
    p.add_argument("--binary", default=os.path.join(ROOT, "target", "release", "iiitd"))
    p.add_argument("--threads", type=int, default=16, help="torch threads per sidecar (0 = no limit)")
    p.add_argument("--timeout", type=int, default=600, help="seconds to wait for every node to be ready")
    args = p.parse_args()
    if args.action == "up":
        up(args)
    elif args.action == "wait":
        sys.exit(0 if wait(args.configs, args.timeout) else 1)
    else:
        down(args)


if __name__ == "__main__":
    main()
