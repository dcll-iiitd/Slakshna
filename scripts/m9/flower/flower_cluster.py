"""Run Flower's FedRAG on its Deployment Runtime: one SuperLink, N SuperNodes.

    python3 scripts/m9/flower/flower_cluster.py up   --data datasets/textbooks --scheme book --nodes 5
    python3 scripts/m9/flower/flower_cluster.py run  --data datasets/textbooks --scheme book \
        --out results/m9/raw/book/flower.jsonl [--qids datasets/textbooks/subset_200.json --warmup 20 --repeat 3]
    python3 scripts/m9/flower/flower_cluster.py down

`up` starts the SuperLink and N SuperNodes as separate processes (each its own
process group; logs and state in m9-flwr/), writes a `[superlink.m9]`
connection into m9-flwr/home/config.toml, and waits until all N SuperNodes
are connected. Every SuperNode gets FEDRAG_CORPUS_DIR=<data>/flower/<scheme>,
so client i searches its own shard's index (the SuperLink gets it too: the
ServerApp checks that the indexes exist). `run` submits the M9 app
(flower-fedrag-m9; --variant stock for Flower's own server with its LLM, check
E0(a)) with `flwr run --stream` and returns when the run ends. Only one
federation runs at a time. Standard library only.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STATE = os.path.join(ROOT, "m9-flwr")
CLI_HOME = os.path.join(STATE, "home")
VENV_BIN = os.path.join(ROOT, ".venv-flower", "bin")
CONNECTION = "m9"


def cli_env():
    env = dict(os.environ, FLWR_HOME=CLI_HOME)
    env["PATH"] = VENV_BIN + os.pathsep + env.get("PATH", "")
    return env


def flwr(*args, check=True, capture=True):
    return subprocess.run([os.path.join(VENV_BIN, "flwr"), *args], env=cli_env(), cwd=ROOT,
                          check=check, capture_output=capture, text=True)


def write_connection(port):
    os.makedirs(CLI_HOME, exist_ok=True)
    flwr("config", "list", check=False)  # creates the default config.toml
    path = os.path.join(CLI_HOME, "config.toml")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    section = f"[superlink.{CONNECTION}]"
    if section in text:
        text = text[:text.index(section)].rstrip() + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + f'\n{section}\naddress = "127.0.0.1:{port}"\ninsecure = true\n')


def connected_supernodes():
    result = flwr("supernode", "list", CONNECTION, "--format", "json", check=False)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    nodes = data.get("nodes", data if isinstance(data, list) else [])
    return [n for n in nodes if str(n.get("status", "online")).lower() in ("online", "connected", "active")]


def start(name, command, env, logs):
    with open(os.path.join(logs, f"{name}.log"), "ab") as log:
        return subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True).pid


def up(args):
    pid_path = os.path.join(STATE, "pids.json")
    if os.path.exists(pid_path):
        sys.exit(f"{pid_path} exists: run 'down' first")
    corpus_dir = os.path.realpath(os.path.join(args.data, "flower", args.scheme))
    for i in range(args.nodes):
        if not os.path.exists(os.path.join(corpus_dir, f"textbooks_s{i}", "faiss.index")):
            sys.exit(f"no index for shard {i} in {corpus_dir}: run build_indexes.sh first")
    logs = os.path.join(STATE, "logs")
    os.makedirs(logs, exist_ok=True)
    write_connection(args.port)

    base = dict(os.environ, PATH=VENV_BIN + os.pathsep + os.environ.get("PATH", ""))
    pids = {"superlink": start("superlink", [
        os.path.join(VENV_BIN, "flower-superlink"), "--insecure", "--isolation", "subprocess",
        "--disable-runtime-dependency-installation", "--host", "127.0.0.1", "--port", str(args.port),
    ], dict(base, FLWR_HOME=os.path.join(STATE, "superlink"), FEDRAG_CORPUS_DIR=corpus_dir), logs)}
    time.sleep(3)
    for i in range(args.nodes):
        env = dict(base, FLWR_HOME=os.path.join(STATE, f"supernode{i}"), FEDRAG_CORPUS_DIR=corpus_dir)
        # Flower's client loads the embedder on every query; online, each load also
        # asks huggingface.co about the model, which can be rate-limited. The
        # model is already cached by build_indexes.sh, so the clients stay offline.
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if args.threads:
            env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        if args.nprobe:
            env["FEDRAG_NPROBE"] = args.nprobe
        pids[f"supernode{i}"] = start(f"supernode{i}", [
            os.path.join(VENV_BIN, "flower-supernode"), "--insecure",
            "--superlink", f"127.0.0.1:{args.port}",
            "--host", "127.0.0.1", "--port", str(args.supernode_port + i),
            "--node-config", f"partition-id={i}",
        ], env, logs)
    with open(pid_path, "w", encoding="utf-8") as f:
        json.dump({"pids": pids, "nodes": args.nodes, "scheme": args.scheme, "data": args.data}, f)

    deadline = time.time() + args.timeout
    while True:
        nodes = connected_supernodes()
        count = len(nodes) if nodes is not None else 0
        if count >= args.nodes:
            print(f"SuperLink and {count} SuperNodes ready (scheme {args.scheme}, corpus {corpus_dir})",
                  file=sys.stderr)
            return
        if time.time() > deadline:
            sys.exit(f"only {count} of {args.nodes} SuperNodes connected after {args.timeout} s; see {logs}")
        print(f"  {count}/{args.nodes} SuperNodes connected", file=sys.stderr)
        time.sleep(3)


def run(args):
    with open(os.path.join(STATE, "pids.json"), encoding="utf-8") as f:
        info = json.load(f)
    if info["scheme"] != args.scheme:
        sys.exit(f"the running federation serves scheme {info['scheme']}, not {args.scheme}")
    app = os.path.join(ROOT, f"flower-fedrag-{args.variant}")
    overrides = {"server-qa-num": args.qa_num}
    if args.variant == "m9":
        overrides.update({
            "m9-questions": os.path.realpath(os.path.join(args.data, "questions.jsonl")),
            "m9-out": os.path.abspath(args.out),
            "m9-qids": os.path.realpath(args.qids) if args.qids else "",
            "m9-system": args.system, "m9-scheme": args.scheme, "m9-nodes": info["nodes"],
            "m9-repeat": args.repeat, "m9-warmup": args.warmup, "m9-llm": "true" if args.llm else "false",
        })
    if args.llm_model:
        overrides["server-llm-hfpath"] = args.llm_model
    toml_path = os.path.join(STATE, "run-config.toml")
    with open(toml_path, "w", encoding="utf-8") as f:
        for key, value in overrides.items():
            f.write(f"{key} = {json.dumps(value)}\n")  # JSON strings and numbers are valid TOML
    before = count_lines(args.out) if args.variant == "m9" else 0
    started = time.time()
    result = flwr("run", app, CONNECTION, "--stream", "--run-config", toml_path, check=False, capture=False)
    print(f"flwr run finished in {time.time() - started:.0f} s (exit {result.returncode})", file=sys.stderr)
    if result.returncode:
        sys.exit(result.returncode)
    if args.variant == "m9":
        # `flwr run --stream` exits 0 even when the ServerApp crashed, so check the output instead.
        added = count_lines(args.out) - before
        print(f"{added} new lines in {args.out}", file=sys.stderr)
        if added == 0:
            sys.exit("the run wrote nothing: see the log above and m9-flwr/logs/")


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def down(args):
    pid_path = os.path.join(STATE, "pids.json")
    try:
        with open(pid_path, encoding="utf-8") as f:
            pids = json.load(f)["pids"]
    except FileNotFoundError:
        print("no pids.json: nothing to stop", file=sys.stderr)
        return
    for pid in pids.values():
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    time.sleep(5)
    for pid in pids.values():
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os.remove(pid_path)
    print(f"stopped {len(pids)} Flower processes", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["up", "run", "down"])
    p.add_argument("--data", help="dataset folder")
    p.add_argument("--scheme")
    p.add_argument("--nodes", type=int, default=5)
    p.add_argument("--port", type=int, default=18000, help="SuperLink port")
    p.add_argument("--supernode-port", type=int, default=18094, help="SuperNode i uses this + i")
    p.add_argument("--threads", type=int, default=16, help="torch threads per client (0 = no limit)")
    p.add_argument("--nprobe", help='IVF buckets each client searches (run E2b; "all" = exact). Default: Flower\'s 1')
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--variant", choices=["m9", "stock"], default="m9")
    p.add_argument("--out", help="contexts file to write (m9 variant)")
    p.add_argument("--qids", help="JSON list of question ids")
    p.add_argument("--qa-num", type=int, default=0, help="first N questions per dataset (0 = all)")
    p.add_argument("--system", default="flower")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--llm", action="store_true", help="also answer inline with Flower's LLMQuerier (E0(a))")
    p.add_argument("--llm-model", help="override server-llm-hfpath")
    args = p.parse_args()
    if args.action in ("up", "run") and not (args.data and args.scheme):
        p.error(f"{args.action} needs --data and --scheme")
    if args.action == "run" and args.variant == "m9" and not args.out:
        p.error("run needs --out")
    {"up": up, "run": run, "down": down}[args.action](args)


if __name__ == "__main__":
    main()
