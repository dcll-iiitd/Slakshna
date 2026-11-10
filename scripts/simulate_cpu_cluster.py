#!/usr/bin/env python
"""
Run a SLAKSHNA federation on ONE machine without the Rust node / network layer.

It reproduces exactly what src/main.rs does each epoch, minus the P2P transport:

  1. stage every peer's most recent delta in <data_dir>/network_deltas/<peer>_delta.b64
  2. launch `python ml_engine.py <me> <peers...>` for all nodes concurrently, with the
     same environment variables the Rust node sets
  3. read the last stdout line (JSON) of each engine and keep its `compressed_delta`
     for the next round

Use it to (a) verify the Python/ML side on a CPU-only laptop before building the Rust
binary, or (b) run the federated training itself when you only care about the ML.

    python scripts/simulate_cpu_cluster.py --config-dir examples/cpu/2node --rounds 3
    python scripts/simulate_cpu_cluster.py --config-dir examples/cpu/4node --rounds 8 --malicious node-2
    python scripts/simulate_cpu_cluster.py --config-dir examples/cpu/2node --reset   # wipe previous run state
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

try:
    import tomllib

    def _load_toml(path):
        with open(path, "rb") as f:
            return tomllib.load(f)
except ModuleNotFoundError:  # Python < 3.11: use the `toml` package from requirements-cpu.txt
    import toml

    def _load_toml(path):
        return toml.load(path)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_nodes(config_dir):
    nodes = []
    for path in sorted(glob.glob(os.path.join(config_dir, "node*.toml"))):
        nodes.append(_load_toml(path))
    if not nodes:
        raise SystemExit(f"No node*.toml found in {config_dir}")
    return nodes


def node_env(cfg, base_env, malicious):
    comp = cfg.get("compression", {})
    train = cfg.get("training", {})
    node = cfg["node"]
    env = dict(base_env)
    env.update({
        "IIITD_DATA_DIR": node["data_dir"],
        "SLAKSHNA_COMPRESSION_ENABLED": str(comp.get("enabled", True)).lower(),
        "SLAKSHNA_DELTA_SPARSITY": str(comp.get("sparsity", 0.1)),
        "SLAKSHNA_DELTA_QUANTIZATION": comp.get("quantization", "symmetric_int8"),
        "SLAKSHNA_ALLOW_LEGACY_DELTA_FORMAT": str(comp.get("allow_legacy_delta_format", False)).lower(),
        "SLAKSHNA_MAX_DELTA_PAYLOAD_BYTES": str(comp.get("max_payload_bytes", 7 * 1024 * 1024)),
        "SLAKSHNA_MAX_DELTA_TENSOR_ELEMENTS": str(comp.get("max_tensor_elements", 10_000_000)),
        "SLAKSHNA_DEVICE": node.get("device", "cpu"),
        "SLAKSHNA_EXPECTED_PEERS": str(train.get("expected_peers", 1)),
        "SLAKSHNA_SYNC_DEADLINE_SECS": str(train.get("sync_deadline_secs", 300)),
        "PYTHONUTF8": "1",
    })
    if node.get("cpu_threads"):
        env["SLAKSHNA_CPU_THREADS"] = str(node["cpu_threads"])
    if malicious:
        env["MALICIOUS_NODES"] = ",".join(malicious)
    return env


def reset(nodes):
    targets = [os.path.join(ROOT, d) for d in ("ml_models", "ml_states")]
    targets += [os.path.join(ROOT, n["node"]["data_dir"]) for n in nodes]
    targets += glob.glob(os.path.join(ROOT, "config_*.yaml"))
    targets += glob.glob(os.path.join(ROOT, "data", "data_*"))
    targets += [os.path.join(ROOT, "logs", f) for f in
                ("epoch_loss_tracking_cpu.csv", "trust_scores_new.csv", "runtime_comm.log")]
    for t in targets:
        if os.path.isdir(t):
            shutil.rmtree(t, ignore_errors=True)
        elif os.path.isfile(t):
            os.remove(t)
    print(f"Reset: removed run state for {len(nodes)} node(s). (Model cache and tokenized cache kept.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=os.path.join(ROOT, "examples", "cpu", "2node"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--malicious", nargs="*", default=[], help="node ids that poison their update, e.g. node-2")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--reset", action="store_true", help="delete previous run state first")
    args = ap.parse_args()

    os.chdir(ROOT)  # ml_engine.py and the TOMLs' relative data_dir assume the repo root
    nodes = load_nodes(args.config_dir)
    ids = [n["node"]["id"] for n in nodes]
    if args.reset:
        reset(nodes)

    os.makedirs("logs", exist_ok=True)
    latest = {}     # node id -> compressed_delta (what the Rust node would have in history)
    print(f"Simulating {len(ids)} node(s): {ids} | rounds={args.rounds} | malicious={args.malicious or 'none'}")

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        for n in nodes:  # 1. stage peers' latest updates
            d = os.path.join(n["node"]["data_dir"], "network_deltas")
            os.makedirs(d, exist_ok=True)
            for peer, delta in latest.items():
                if peer != n["node"]["id"]:
                    with open(os.path.join(d, f"{peer}_delta.b64"), "w") as f:
                        f.write(delta)

        procs = {}  # 2. launch all engines concurrently
        for n in nodes:
            me = n["node"]["id"]
            neighbors = [p for p in latest if p != me]     # Rust passes peers that have records
            log = open(os.path.join("logs", f"sim_{me}.log"), "a", encoding="utf-8")
            log.write(f"\n===== round {rnd} =====\n")
            log.flush()
            procs[me] = (
                subprocess.Popen(
                    [args.python, "ml_engine.py", me, *neighbors],
                    stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                    env=node_env(n, os.environ, args.malicious),
                ),
                log,
            )
        print(f"\n--- Round {rnd}/{args.rounds}: {len(procs)} engine(s) running (logs/sim_<id>.log) ---", flush=True)

        finished = False
        for me, (p, log) in procs.items():  # 3. collect
            out, _ = p.communicate()
            log.close()
            if p.returncode != 0:
                print(f"  {me}: FAILED (exit {p.returncode}) - see logs/sim_{me}.log")
                continue
            try:
                res = json.loads(out.strip().splitlines()[-1])
            except Exception:
                print(f"  {me}: no JSON result - see logs/sim_{me}.log")
                continue
            latest[me] = res["compressed_delta"]
            finished = finished or res.get("is_finished", False)
            w = ", ".join(f"{k}:{v:.2f}" for k, v in sorted(res["weights"].items()))
            print(f"  {me}: loss={res['validation_score']:.4f}  trust[{w}]  delta={len(res['compressed_delta'])/1024:.0f} KiB")
        print(f"  round time: {time.time() - t0:.0f}s")
        if finished:
            print("Target number of federated rounds reached.")
            break


if __name__ == "__main__":
    main()
